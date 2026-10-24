import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root325Candidates : List Nat := Data.row2203
set_option maxRecDepth 8192 in
theorem root325Candidates_checked : candidates edge 5 0 2203 tail326 winner.bound = root325Candidates := by
  apply candidates_eq_row edge 5 0 2203 tail326 root325Candidates winner.bound tail326_ordered
  · decide
  · decide
  · funext q
    rfl
def root325Tail43 : List Nat := []
def root325Tail42 : List Nat := 23071 :: root325Tail43
def root325Tail41 : List Nat := 22573 :: root325Tail42
def root325Tail40 : List Nat := 21589 :: root325Tail41
def root325Tail39 : List Nat := 21559 :: root325Tail40
def root325Tail38 : List Nat := 21523 :: root325Tail39
def root325Tail37 : List Nat := 21499 :: root325Tail38
def root325Tail36 : List Nat := 21187 :: root325Tail37
def root325Tail35 : List Nat := 20407 :: root325Tail36
def root325Tail34 : List Nat := 19483 :: root325Tail35
def root325Tail33 : List Nat := 19207 :: root325Tail34
def root325Tail32 : List Nat := 18517 :: root325Tail33
def root325Tail31 : List Nat := 18493 :: root325Tail32
def root325Tail30 : List Nat := 17599 :: root325Tail31
def root325Tail29 : List Nat := 17389 :: root325Tail30
def root325Tail28 : List Nat := 16273 :: root325Tail29
def root325Tail27 : List Nat := 14821 :: root325Tail28
def root325Tail26 : List Nat := 14449 :: root325Tail27
def root325Tail25 : List Nat := 14251 :: root325Tail26
def root325Tail24 : List Nat := 12487 :: root325Tail25
def root325Tail23 : List Nat := 12409 :: root325Tail24
def root325Tail22 : List Nat := 12241 :: root325Tail23
def root325Tail21 : List Nat := 11863 :: root325Tail22
def root325Tail20 : List Nat := 11353 :: root325Tail21
def root325Tail19 : List Nat := 10243 :: root325Tail20
def root325Tail18 : List Nat := 10141 :: root325Tail19
def root325Tail17 : List Nat := 9697 :: root325Tail18
def root325Tail16 : List Nat := 9133 :: root325Tail17
def root325Tail15 : List Nat := 8389 :: root325Tail16
def root325Tail14 : List Nat := 8161 :: root325Tail15
def root325Tail13 : List Nat := 7927 :: root325Tail14
def root325Tail12 : List Nat := 7789 :: root325Tail13
def root325Tail11 : List Nat := 6961 :: root325Tail12
def root325Tail10 : List Nat := 6529 :: root325Tail11
def root325Tail9 : List Nat := 6271 :: root325Tail10
def root325Tail8 : List Nat := 6037 :: root325Tail9
def root325Tail7 : List Nat := 5857 :: root325Tail8
def root325Tail6 : List Nat := 5647 :: root325Tail7
def root325Tail5 : List Nat := 5623 :: root325Tail6
def root325Tail4 : List Nat := 5557 :: root325Tail5
def root325Tail3 : List Nat := 5521 :: root325Tail4
def root325Tail2 : List Nat := 3061 :: root325Tail3
def root325Tail1 : List Nat := 2887 :: root325Tail2
def root325Tail0 : List Nat := 2371 :: root325Tail1
def root325Candidates0 : List Nat := [10141]
set_option maxRecDepth 8192 in
theorem root325Candidates0_checked : candidates edge 4 2203 2371 root325Tail1 winner.bound = root325Candidates0 := by
  unfold candidates
  have row : edge 2371 = (SortedNatTree.contains FastRows.row2371) := by
    change Data.row2371.contains = (SortedNatTree.contains FastRows.row2371)
    exact FastRows.row2371_checked.symm
  rw [row]
  decide
theorem root325Step0 : scan edge (search edge 3) 4 [2203] 2203 root325Tail0 winner =
    scan edge (search edge 3) 4 [2203] 2203 root325Tail1 winner := by
  rw [root325Tail0, scan, if_neg (by decide), root325Candidates0_checked]
  change scan edge (search edge 3) 4 [2203] 2203 root325Tail1
    (if 3 ≤ root325Candidates0.length then search edge 3 [2203, 2371] root325Candidates0 4574 winner else winner) = _
  rw [if_neg (by decide)]
def root325Candidates1 : List Nat := [14821]
set_option maxRecDepth 8192 in
theorem root325Candidates1_checked : candidates edge 4 2203 2887 root325Tail2 winner.bound = root325Candidates1 := by
  unfold candidates
  have row : edge 2887 = (SortedNatTree.contains FastRows.row2887) := by
    change Data.row2887.contains = (SortedNatTree.contains FastRows.row2887)
    exact FastRows.row2887_checked.symm
  rw [row]
  decide
theorem root325Step1 : scan edge (search edge 3) 4 [2203] 2203 root325Tail1 winner =
    scan edge (search edge 3) 4 [2203] 2203 root325Tail2 winner := by
  rw [root325Tail1, scan, if_neg (by decide), root325Candidates1_checked]
  change scan edge (search edge 3) 4 [2203] 2203 root325Tail2
    (if 3 ≤ root325Candidates1.length then search edge 3 [2203, 2887] root325Candidates1 5090 winner else winner) = _
  rw [if_neg (by decide)]
def root325Candidates2 : List Nat := [5857]
set_option maxRecDepth 8192 in
theorem root325Candidates2_checked : candidates edge 4 2203 3061 root325Tail3 winner.bound = root325Candidates2 := by
  unfold candidates
  have row : edge 3061 = (SortedNatTree.contains FastRows.row3061) := by
    change Data.row3061.contains = (SortedNatTree.contains FastRows.row3061)
    exact FastRows.row3061_checked.symm
  rw [row]
  decide
theorem root325Step2 : scan edge (search edge 3) 4 [2203] 2203 root325Tail2 winner =
    scan edge (search edge 3) 4 [2203] 2203 root325Tail3 winner := by
  rw [root325Tail2, scan, if_neg (by decide), root325Candidates2_checked]
  change scan edge (search edge 3) 4 [2203] 2203 root325Tail3
    (if 3 ≤ root325Candidates2.length then search edge 3 [2203, 3061] root325Candidates2 5264 winner else winner) = _
  rw [if_neg (by decide)]
def root325Candidates3 : List Nat := [12487, 14821]
set_option maxRecDepth 8192 in
theorem root325Candidates3_checked : candidates edge 4 2203 5521 root325Tail4 winner.bound = root325Candidates3 := by
  unfold candidates
  have row : edge 5521 = (SortedNatTree.contains FastRows.row5521) := by
    change Data.row5521.contains = (SortedNatTree.contains FastRows.row5521)
    exact FastRows.row5521_checked.symm
  rw [row]
  decide
theorem root325Step3 : scan edge (search edge 3) 4 [2203] 2203 root325Tail3 winner =
    scan edge (search edge 3) 4 [2203] 2203 root325Tail4 winner := by
  rw [root325Tail3, scan, if_neg (by decide), root325Candidates3_checked]
  change scan edge (search edge 3) 4 [2203] 2203 root325Tail4
    (if 3 ≤ root325Candidates3.length then search edge 3 [2203, 5521] root325Candidates3 7724 winner else winner) = _
  rw [if_neg (by decide)]
def root325Candidates4 : List Nat := [6529, 7789, 9133]
set_option maxRecDepth 8192 in
theorem root325Candidates4_checked : candidates edge 4 2203 5557 root325Tail5 winner.bound = root325Candidates4 := by
  unfold candidates
  have row : edge 5557 = (SortedNatTree.contains FastRows.row5557) := by
    change Data.row5557.contains = (SortedNatTree.contains FastRows.row5557)
    exact FastRows.row5557_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root325Child4 : search edge 3 [2203, 5557] root325Candidates4 7760 winner = winner := by decide
theorem root325Step4 : scan edge (search edge 3) 4 [2203] 2203 root325Tail4 winner =
    scan edge (search edge 3) 4 [2203] 2203 root325Tail5 winner := by
  rw [root325Tail4, scan, if_neg (by decide), root325Candidates4_checked]
  change scan edge (search edge 3) 4 [2203] 2203 root325Tail5
    (if 3 ≤ root325Candidates4.length then search edge 3 [2203, 5557] root325Candidates4 7760 winner else winner) = _
  rw [if_pos (by decide), root325Child4]
def root325Candidates5 : List Nat := [11353, 14449]
set_option maxRecDepth 8192 in
theorem root325Candidates5_checked : candidates edge 4 2203 5623 root325Tail6 winner.bound = root325Candidates5 := by
  unfold candidates
  have row : edge 5623 = (SortedNatTree.contains FastRows.row5623) := by
    change Data.row5623.contains = (SortedNatTree.contains FastRows.row5623)
    exact FastRows.row5623_checked.symm
  rw [row]
  decide
theorem root325Step5 : scan edge (search edge 3) 4 [2203] 2203 root325Tail5 winner =
    scan edge (search edge 3) 4 [2203] 2203 root325Tail6 winner := by
  rw [root325Tail5, scan, if_neg (by decide), root325Candidates5_checked]
  change scan edge (search edge 3) 4 [2203] 2203 root325Tail6
    (if 3 ≤ root325Candidates5.length then search edge 3 [2203, 5623] root325Candidates5 7826 winner else winner) = _
  rw [if_neg (by decide)]
def root325Candidates6 : List Nat := [10141]
set_option maxRecDepth 8192 in
theorem root325Candidates6_checked : candidates edge 4 2203 5647 root325Tail7 winner.bound = root325Candidates6 := by
  unfold candidates
  have row : edge 5647 = (SortedNatTree.contains FastRows.row5647) := by
    change Data.row5647.contains = (SortedNatTree.contains FastRows.row5647)
    exact FastRows.row5647_checked.symm
  rw [row]
  decide
theorem root325Step6 : scan edge (search edge 3) 4 [2203] 2203 root325Tail6 winner =
    scan edge (search edge 3) 4 [2203] 2203 root325Tail7 winner := by
  rw [root325Tail6, scan, if_neg (by decide), root325Candidates6_checked]
  change scan edge (search edge 3) 4 [2203] 2203 root325Tail7
    (if 3 ≤ root325Candidates6.length then search edge 3 [2203, 5647] root325Candidates6 7850 winner else winner) = _
  rw [if_neg (by decide)]
def root325Candidates7 : List Nat := []
set_option maxRecDepth 8192 in
theorem root325Candidates7_checked : candidates edge 4 2203 5857 root325Tail8 winner.bound = root325Candidates7 := by
  unfold candidates
  have row : edge 5857 = (SortedNatTree.contains FastRows.row5857) := by
    change Data.row5857.contains = (SortedNatTree.contains FastRows.row5857)
    exact FastRows.row5857_checked.symm
  rw [row]
  decide
theorem root325Step7 : scan edge (search edge 3) 4 [2203] 2203 root325Tail7 winner =
    scan edge (search edge 3) 4 [2203] 2203 root325Tail8 winner := by
  rw [root325Tail7, scan, if_neg (by decide), root325Candidates7_checked]
  change scan edge (search edge 3) 4 [2203] 2203 root325Tail8
    (if 3 ≤ root325Candidates7.length then search edge 3 [2203, 5857] root325Candidates7 8060 winner else winner) = _
  rw [if_neg (by decide)]
theorem root325Scan8 : scan edge (search edge 3) 4 [2203] 2203 root325Tail8 winner = winner := by rw [root325Tail8, scan, if_pos (by decide)]
theorem root325Scan7 : scan edge (search edge 3) 4 [2203] 2203 root325Tail7 winner = winner :=
  (root325Step7).trans root325Scan8
theorem root325Scan6 : scan edge (search edge 3) 4 [2203] 2203 root325Tail6 winner = winner :=
  (root325Step6).trans root325Scan7
theorem root325Scan5 : scan edge (search edge 3) 4 [2203] 2203 root325Tail5 winner = winner :=
  (root325Step5).trans root325Scan6
theorem root325Scan4 : scan edge (search edge 3) 4 [2203] 2203 root325Tail4 winner = winner :=
  (root325Step4).trans root325Scan5
theorem root325Scan3 : scan edge (search edge 3) 4 [2203] 2203 root325Tail3 winner = winner :=
  (root325Step3).trans root325Scan4
theorem root325Scan2 : scan edge (search edge 3) 4 [2203] 2203 root325Tail2 winner = winner :=
  (root325Step2).trans root325Scan3
theorem root325Scan1 : scan edge (search edge 3) 4 [2203] 2203 root325Tail1 winner = winner :=
  (root325Step1).trans root325Scan2
theorem root325Scan0 : scan edge (search edge 3) 4 [2203] 2203 root325Tail0 winner = winner :=
  (root325Step0).trans root325Scan1
set_option maxRecDepth 4096 in
theorem root325Branch : search edge 4 [2203] root325Candidates 2203 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2203] 2203 root325Tail0 winner = winner
  exact root325Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
