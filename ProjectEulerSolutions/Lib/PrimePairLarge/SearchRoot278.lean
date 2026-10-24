import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root278Candidates : List Nat := Data.row1823
set_option maxRecDepth 8192 in
theorem root278Candidates_checked : candidates edge 5 0 1823 tail279 winner.bound = root278Candidates := by
  apply candidates_eq_row edge 5 0 1823 tail279 root278Candidates winner.bound tail279_ordered
  · decide
  · decide
  · funext q
    rfl
def root278Tail42 : List Nat := []
def root278Tail41 : List Nat := 24029 :: root278Tail42
def root278Tail40 : List Nat := 23993 :: root278Tail41
def root278Tail39 : List Nat := 23117 :: root278Tail40
def root278Tail38 : List Nat := 21767 :: root278Tail39
def root278Tail37 : List Nat := 19469 :: root278Tail38
def root278Tail36 : List Nat := 19001 :: root278Tail37
def root278Tail35 : List Nat := 18671 :: root278Tail36
def root278Tail34 : List Nat := 17957 :: root278Tail35
def root278Tail33 : List Nat := 17189 :: root278Tail34
def root278Tail32 : List Nat := 16097 :: root278Tail33
def root278Tail31 : List Nat := 15629 :: root278Tail32
def root278Tail30 : List Nat := 14843 :: root278Tail31
def root278Tail29 : List Nat := 14669 :: root278Tail30
def root278Tail28 : List Nat := 14423 :: root278Tail29
def root278Tail27 : List Nat := 14159 :: root278Tail28
def root278Tail26 : List Nat := 13997 :: root278Tail27
def root278Tail25 : List Nat := 12659 :: root278Tail26
def root278Tail24 : List Nat := 12203 :: root278Tail25
def root278Tail23 : List Nat := 12161 :: root278Tail24
def root278Tail22 : List Nat := 12149 :: root278Tail23
def root278Tail21 : List Nat := 11903 :: root278Tail22
def root278Tail20 : List Nat := 10859 :: root278Tail21
def root278Tail19 : List Nat := 10853 :: root278Tail20
def root278Tail18 : List Nat := 10631 :: root278Tail19
def root278Tail17 : List Nat := 9341 :: root278Tail18
def root278Tail16 : List Nat := 8783 :: root278Tail17
def root278Tail15 : List Nat := 8069 :: root278Tail16
def root278Tail14 : List Nat := 7823 :: root278Tail15
def root278Tail13 : List Nat := 6983 :: root278Tail14
def root278Tail12 : List Nat := 6947 :: root278Tail13
def root278Tail11 : List Nat := 6719 :: root278Tail12
def root278Tail10 : List Nat := 6353 :: root278Tail11
def root278Tail9 : List Nat := 6131 :: root278Tail10
def root278Tail8 : List Nat := 4493 :: root278Tail9
def root278Tail7 : List Nat := 4001 :: root278Tail8
def root278Tail6 : List Nat := 3947 :: root278Tail7
def root278Tail5 : List Nat := 3461 :: root278Tail6
def root278Tail4 : List Nat := 3023 :: root278Tail5
def root278Tail3 : List Nat := 2861 :: root278Tail4
def root278Tail2 : List Nat := 2111 :: root278Tail3
def root278Tail1 : List Nat := 2069 :: root278Tail2
def root278Tail0 : List Nat := 2063 :: root278Tail1
def root278Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root278Candidates0_checked : candidates edge 4 1823 2063 root278Tail1 winner.bound = root278Candidates0 := by
  unfold candidates
  have row : edge 2063 = (SortedNatTree.contains FastRows.row2063) := by
    change Data.row2063.contains = (SortedNatTree.contains FastRows.row2063)
    exact FastRows.row2063_checked.symm
  rw [row]
  decide
theorem root278Step0 : scan edge (search edge 3) 4 [1823] 1823 root278Tail0 winner =
    scan edge (search edge 3) 4 [1823] 1823 root278Tail1 winner := by
  rw [root278Tail0, scan, if_neg (by decide), root278Candidates0_checked]
  change scan edge (search edge 3) 4 [1823] 1823 root278Tail1
    (if 3 ≤ root278Candidates0.length then search edge 3 [1823, 2063] root278Candidates0 3886 winner else winner) = _
  rw [if_neg (by decide)]
def root278Candidates1 : List Nat := [6983, 14423]
set_option maxRecDepth 8192 in
theorem root278Candidates1_checked : candidates edge 4 1823 2069 root278Tail2 winner.bound = root278Candidates1 := by
  unfold candidates
  have row : edge 2069 = (SortedNatTree.contains FastRows.row2069) := by
    change Data.row2069.contains = (SortedNatTree.contains FastRows.row2069)
    exact FastRows.row2069_checked.symm
  rw [row]
  decide
theorem root278Step1 : scan edge (search edge 3) 4 [1823] 1823 root278Tail1 winner =
    scan edge (search edge 3) 4 [1823] 1823 root278Tail2 winner := by
  rw [root278Tail1, scan, if_neg (by decide), root278Candidates1_checked]
  change scan edge (search edge 3) 4 [1823] 1823 root278Tail2
    (if 3 ≤ root278Candidates1.length then search edge 3 [1823, 2069] root278Candidates1 3892 winner else winner) = _
  rw [if_neg (by decide)]
def root278Candidates2 : List Nat := [14843, 17957]
set_option maxRecDepth 8192 in
theorem root278Candidates2_checked : candidates edge 4 1823 2111 root278Tail3 winner.bound = root278Candidates2 := by
  unfold candidates
  have row : edge 2111 = (SortedNatTree.contains FastRows.row2111) := by
    change Data.row2111.contains = (SortedNatTree.contains FastRows.row2111)
    exact FastRows.row2111_checked.symm
  rw [row]
  decide
theorem root278Step2 : scan edge (search edge 3) 4 [1823] 1823 root278Tail2 winner =
    scan edge (search edge 3) 4 [1823] 1823 root278Tail3 winner := by
  rw [root278Tail2, scan, if_neg (by decide), root278Candidates2_checked]
  change scan edge (search edge 3) 4 [1823] 1823 root278Tail3
    (if 3 ≤ root278Candidates2.length then search edge 3 [1823, 2111] root278Candidates2 3934 winner else winner) = _
  rw [if_neg (by decide)]
def root278Candidates3 : List Nat := [6719, 12161]
set_option maxRecDepth 8192 in
theorem root278Candidates3_checked : candidates edge 4 1823 2861 root278Tail4 winner.bound = root278Candidates3 := by
  unfold candidates
  have row : edge 2861 = (SortedNatTree.contains FastRows.row2861) := by
    change Data.row2861.contains = (SortedNatTree.contains FastRows.row2861)
    exact FastRows.row2861_checked.symm
  rw [row]
  decide
theorem root278Step3 : scan edge (search edge 3) 4 [1823] 1823 root278Tail3 winner =
    scan edge (search edge 3) 4 [1823] 1823 root278Tail4 winner := by
  rw [root278Tail3, scan, if_neg (by decide), root278Candidates3_checked]
  change scan edge (search edge 3) 4 [1823] 1823 root278Tail4
    (if 3 ≤ root278Candidates3.length then search edge 3 [1823, 2861] root278Candidates3 4684 winner else winner) = _
  rw [if_neg (by decide)]
def root278Candidates4 : List Nat := [8069, 8783, 19001]
set_option maxRecDepth 8192 in
theorem root278Candidates4_checked : candidates edge 4 1823 3023 root278Tail5 winner.bound = root278Candidates4 := by
  unfold candidates
  have row : edge 3023 = (SortedNatTree.contains FastRows.row3023) := by
    change Data.row3023.contains = (SortedNatTree.contains FastRows.row3023)
    exact FastRows.row3023_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root278Child4 : search edge 3 [1823, 3023] root278Candidates4 4846 winner = winner := by decide
theorem root278Step4 : scan edge (search edge 3) 4 [1823] 1823 root278Tail4 winner =
    scan edge (search edge 3) 4 [1823] 1823 root278Tail5 winner := by
  rw [root278Tail4, scan, if_neg (by decide), root278Candidates4_checked]
  change scan edge (search edge 3) 4 [1823] 1823 root278Tail5
    (if 3 ≤ root278Candidates4.length then search edge 3 [1823, 3023] root278Candidates4 4846 winner else winner) = _
  rw [if_pos (by decide), root278Child4]
def root278Candidates5 : List Nat := [7823, 14423]
set_option maxRecDepth 8192 in
theorem root278Candidates5_checked : candidates edge 4 1823 3461 root278Tail6 winner.bound = root278Candidates5 := by
  unfold candidates
  have row : edge 3461 = (SortedNatTree.contains FastRows.row3461) := by
    change Data.row3461.contains = (SortedNatTree.contains FastRows.row3461)
    exact FastRows.row3461_checked.symm
  rw [row]
  decide
theorem root278Step5 : scan edge (search edge 3) 4 [1823] 1823 root278Tail5 winner =
    scan edge (search edge 3) 4 [1823] 1823 root278Tail6 winner := by
  rw [root278Tail5, scan, if_neg (by decide), root278Candidates5_checked]
  change scan edge (search edge 3) 4 [1823] 1823 root278Tail6
    (if 3 ≤ root278Candidates5.length then search edge 3 [1823, 3461] root278Candidates5 5284 winner else winner) = _
  rw [if_neg (by decide)]
def root278Candidates6 : List Nat := []
set_option maxRecDepth 8192 in
theorem root278Candidates6_checked : candidates edge 4 1823 3947 root278Tail7 winner.bound = root278Candidates6 := by
  unfold candidates
  have row : edge 3947 = (SortedNatTree.contains FastRows.row3947) := by
    change Data.row3947.contains = (SortedNatTree.contains FastRows.row3947)
    exact FastRows.row3947_checked.symm
  rw [row]
  decide
theorem root278Step6 : scan edge (search edge 3) 4 [1823] 1823 root278Tail6 winner =
    scan edge (search edge 3) 4 [1823] 1823 root278Tail7 winner := by
  rw [root278Tail6, scan, if_neg (by decide), root278Candidates6_checked]
  change scan edge (search edge 3) 4 [1823] 1823 root278Tail7
    (if 3 ≤ root278Candidates6.length then search edge 3 [1823, 3947] root278Candidates6 5770 winner else winner) = _
  rw [if_neg (by decide)]
def root278Candidates7 : List Nat := [15629]
set_option maxRecDepth 8192 in
theorem root278Candidates7_checked : candidates edge 4 1823 4001 root278Tail8 winner.bound = root278Candidates7 := by
  unfold candidates
  have row : edge 4001 = (SortedNatTree.contains FastRows.row4001) := by
    change Data.row4001.contains = (SortedNatTree.contains FastRows.row4001)
    exact FastRows.row4001_checked.symm
  rw [row]
  decide
theorem root278Step7 : scan edge (search edge 3) 4 [1823] 1823 root278Tail7 winner =
    scan edge (search edge 3) 4 [1823] 1823 root278Tail8 winner := by
  rw [root278Tail7, scan, if_neg (by decide), root278Candidates7_checked]
  change scan edge (search edge 3) 4 [1823] 1823 root278Tail8
    (if 3 ≤ root278Candidates7.length then search edge 3 [1823, 4001] root278Candidates7 5824 winner else winner) = _
  rw [if_neg (by decide)]
def root278Candidates8 : List Nat := [14843]
set_option maxRecDepth 8192 in
theorem root278Candidates8_checked : candidates edge 4 1823 4493 root278Tail9 winner.bound = root278Candidates8 := by
  unfold candidates
  have row : edge 4493 = (SortedNatTree.contains FastRows.row4493) := by
    change Data.row4493.contains = (SortedNatTree.contains FastRows.row4493)
    exact FastRows.row4493_checked.symm
  rw [row]
  decide
theorem root278Step8 : scan edge (search edge 3) 4 [1823] 1823 root278Tail8 winner =
    scan edge (search edge 3) 4 [1823] 1823 root278Tail9 winner := by
  rw [root278Tail8, scan, if_neg (by decide), root278Candidates8_checked]
  change scan edge (search edge 3) 4 [1823] 1823 root278Tail9
    (if 3 ≤ root278Candidates8.length then search edge 3 [1823, 4493] root278Candidates8 6316 winner else winner) = _
  rw [if_neg (by decide)]
theorem root278Scan9 : scan edge (search edge 3) 4 [1823] 1823 root278Tail9 winner = winner := by rw [root278Tail9, scan, if_pos (by decide)]
theorem root278Scan8 : scan edge (search edge 3) 4 [1823] 1823 root278Tail8 winner = winner :=
  (root278Step8).trans root278Scan9
theorem root278Scan7 : scan edge (search edge 3) 4 [1823] 1823 root278Tail7 winner = winner :=
  (root278Step7).trans root278Scan8
theorem root278Scan6 : scan edge (search edge 3) 4 [1823] 1823 root278Tail6 winner = winner :=
  (root278Step6).trans root278Scan7
theorem root278Scan5 : scan edge (search edge 3) 4 [1823] 1823 root278Tail5 winner = winner :=
  (root278Step5).trans root278Scan6
theorem root278Scan4 : scan edge (search edge 3) 4 [1823] 1823 root278Tail4 winner = winner :=
  (root278Step4).trans root278Scan5
theorem root278Scan3 : scan edge (search edge 3) 4 [1823] 1823 root278Tail3 winner = winner :=
  (root278Step3).trans root278Scan4
theorem root278Scan2 : scan edge (search edge 3) 4 [1823] 1823 root278Tail2 winner = winner :=
  (root278Step2).trans root278Scan3
theorem root278Scan1 : scan edge (search edge 3) 4 [1823] 1823 root278Tail1 winner = winner :=
  (root278Step1).trans root278Scan2
theorem root278Scan0 : scan edge (search edge 3) 4 [1823] 1823 root278Tail0 winner = winner :=
  (root278Step0).trans root278Scan1
set_option maxRecDepth 4096 in
theorem root278Branch : search edge 4 [1823] root278Candidates 1823 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1823] 1823 root278Tail0 winner = winner
  exact root278Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
