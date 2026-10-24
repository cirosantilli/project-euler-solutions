import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root466Candidates : List Nat := Data.row3329
set_option maxRecDepth 8192 in
theorem root466Candidates_checked : candidates edge 5 0 3329 tail467 winner.bound = root466Candidates := by
  apply candidates_eq_row edge 5 0 3329 tail467 root466Candidates winner.bound tail467_ordered
  · decide
  · decide
  · funext q
    rfl
def root466Tail34 : List Nat := []
def root466Tail33 : List Nat := 22307 :: root466Tail34
def root466Tail32 : List Nat := 22229 :: root466Tail33
def root466Tail31 : List Nat := 21101 :: root466Tail32
def root466Tail30 : List Nat := 20879 :: root466Tail31
def root466Tail29 : List Nat := 18743 :: root466Tail30
def root466Tail28 : List Nat := 18587 :: root466Tail29
def root466Tail27 : List Nat := 18041 :: root466Tail28
def root466Tail26 : List Nat := 16931 :: root466Tail27
def root466Tail25 : List Nat := 16487 :: root466Tail26
def root466Tail24 : List Nat := 16091 :: root466Tail25
def root466Tail23 : List Nat := 15629 :: root466Tail24
def root466Tail22 : List Nat := 15263 :: root466Tail23
def root466Tail21 : List Nat := 15137 :: root466Tail22
def root466Tail20 : List Nat := 13967 :: root466Tail21
def root466Tail19 : List Nat := 12491 :: root466Tail20
def root466Tail18 : List Nat := 12119 :: root466Tail19
def root466Tail17 : List Nat := 11519 :: root466Tail18
def root466Tail16 : List Nat := 11351 :: root466Tail17
def root466Tail15 : List Nat := 10559 :: root466Tail16
def root466Tail14 : List Nat := 10181 :: root466Tail15
def root466Tail13 : List Nat := 9887 :: root466Tail14
def root466Tail12 : List Nat := 9437 :: root466Tail13
def root466Tail11 : List Nat := 8627 :: root466Tail12
def root466Tail10 : List Nat := 7877 :: root466Tail11
def root466Tail9 : List Nat := 7109 :: root466Tail10
def root466Tail8 : List Nat := 7001 :: root466Tail9
def root466Tail7 : List Nat := 6359 :: root466Tail8
def root466Tail6 : List Nat := 5441 :: root466Tail7
def root466Tail5 : List Nat := 5051 :: root466Tail6
def root466Tail4 : List Nat := 4967 :: root466Tail5
def root466Tail3 : List Nat := 4649 :: root466Tail4
def root466Tail2 : List Nat := 4637 :: root466Tail3
def root466Tail1 : List Nat := 4127 :: root466Tail2
def root466Tail0 : List Nat := 3929 :: root466Tail1
def root466Candidates0 : List Nat := [7109]
set_option maxRecDepth 8192 in
theorem root466Candidates0_checked : candidates edge 4 3329 3929 root466Tail1 winner.bound = root466Candidates0 := by
  unfold candidates
  have row : edge 3929 = (SortedNatTree.contains FastRows.row3929) := by
    change Data.row3929.contains = (SortedNatTree.contains FastRows.row3929)
    exact FastRows.row3929_checked.symm
  rw [row]
  decide
theorem root466Step0 : scan edge (search edge 3) 4 [3329] 3329 root466Tail0 winner =
    scan edge (search edge 3) 4 [3329] 3329 root466Tail1 winner := by
  rw [root466Tail0, scan, if_neg (by decide), root466Candidates0_checked]
  change scan edge (search edge 3) 4 [3329] 3329 root466Tail1
    (if 3 ≤ root466Candidates0.length then search edge 3 [3329, 3929] root466Candidates0 7258 winner else winner) = _
  rw [if_neg (by decide)]
def root466Candidates1 : List Nat := [4967]
set_option maxRecDepth 8192 in
theorem root466Candidates1_checked : candidates edge 4 3329 4127 root466Tail2 winner.bound = root466Candidates1 := by
  unfold candidates
  have row : edge 4127 = (SortedNatTree.contains FastRows.row4127) := by
    change Data.row4127.contains = (SortedNatTree.contains FastRows.row4127)
    exact FastRows.row4127_checked.symm
  rw [row]
  decide
theorem root466Step1 : scan edge (search edge 3) 4 [3329] 3329 root466Tail1 winner =
    scan edge (search edge 3) 4 [3329] 3329 root466Tail2 winner := by
  rw [root466Tail1, scan, if_neg (by decide), root466Candidates1_checked]
  change scan edge (search edge 3) 4 [3329] 3329 root466Tail2
    (if 3 ≤ root466Candidates1.length then search edge 3 [3329, 4127] root466Candidates1 7456 winner else winner) = _
  rw [if_neg (by decide)]
def root466Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root466Candidates2_checked : candidates edge 4 3329 4637 root466Tail3 winner.bound = root466Candidates2 := by
  unfold candidates
  have row : edge 4637 = (SortedNatTree.contains FastRows.row4637) := by
    change Data.row4637.contains = (SortedNatTree.contains FastRows.row4637)
    exact FastRows.row4637_checked.symm
  rw [row]
  decide
theorem root466Step2 : scan edge (search edge 3) 4 [3329] 3329 root466Tail2 winner =
    scan edge (search edge 3) 4 [3329] 3329 root466Tail3 winner := by
  rw [root466Tail2, scan, if_neg (by decide), root466Candidates2_checked]
  change scan edge (search edge 3) 4 [3329] 3329 root466Tail3
    (if 3 ≤ root466Candidates2.length then search edge 3 [3329, 4637] root466Candidates2 7966 winner else winner) = _
  rw [if_neg (by decide)]
def root466Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root466Candidates3_checked : candidates edge 4 3329 4649 root466Tail4 winner.bound = root466Candidates3 := by
  unfold candidates
  have row : edge 4649 = (SortedNatTree.contains FastRows.row4649) := by
    change Data.row4649.contains = (SortedNatTree.contains FastRows.row4649)
    exact FastRows.row4649_checked.symm
  rw [row]
  decide
theorem root466Step3 : scan edge (search edge 3) 4 [3329] 3329 root466Tail3 winner =
    scan edge (search edge 3) 4 [3329] 3329 root466Tail4 winner := by
  rw [root466Tail3, scan, if_neg (by decide), root466Candidates3_checked]
  change scan edge (search edge 3) 4 [3329] 3329 root466Tail4
    (if 3 ≤ root466Candidates3.length then search edge 3 [3329, 4649] root466Candidates3 7978 winner else winner) = _
  rw [if_neg (by decide)]
def root466Candidates4 : List Nat := [5441, 7877]
set_option maxRecDepth 8192 in
theorem root466Candidates4_checked : candidates edge 4 3329 4967 root466Tail5 winner.bound = root466Candidates4 := by
  unfold candidates
  have row : edge 4967 = (SortedNatTree.contains FastRows.row4967) := by
    change Data.row4967.contains = (SortedNatTree.contains FastRows.row4967)
    exact FastRows.row4967_checked.symm
  rw [row]
  decide
theorem root466Step4 : scan edge (search edge 3) 4 [3329] 3329 root466Tail4 winner =
    scan edge (search edge 3) 4 [3329] 3329 root466Tail5 winner := by
  rw [root466Tail4, scan, if_neg (by decide), root466Candidates4_checked]
  change scan edge (search edge 3) 4 [3329] 3329 root466Tail5
    (if 3 ≤ root466Candidates4.length then search edge 3 [3329, 4967] root466Candidates4 8296 winner else winner) = _
  rw [if_neg (by decide)]
def root466Candidates5 : List Nat := [15137]
set_option maxRecDepth 8192 in
theorem root466Candidates5_checked : candidates edge 4 3329 5051 root466Tail6 winner.bound = root466Candidates5 := by
  unfold candidates
  have row : edge 5051 = (SortedNatTree.contains FastRows.row5051) := by
    change Data.row5051.contains = (SortedNatTree.contains FastRows.row5051)
    exact FastRows.row5051_checked.symm
  rw [row]
  decide
theorem root466Step5 : scan edge (search edge 3) 4 [3329] 3329 root466Tail5 winner =
    scan edge (search edge 3) 4 [3329] 3329 root466Tail6 winner := by
  rw [root466Tail5, scan, if_neg (by decide), root466Candidates5_checked]
  change scan edge (search edge 3) 4 [3329] 3329 root466Tail6
    (if 3 ≤ root466Candidates5.length then search edge 3 [3329, 5051] root466Candidates5 8380 winner else winner) = _
  rw [if_neg (by decide)]
def root466Candidates6 : List Nat := []
set_option maxRecDepth 8192 in
theorem root466Candidates6_checked : candidates edge 4 3329 5441 root466Tail7 winner.bound = root466Candidates6 := by
  unfold candidates
  have row : edge 5441 = (SortedNatTree.contains FastRows.row5441) := by
    change Data.row5441.contains = (SortedNatTree.contains FastRows.row5441)
    exact FastRows.row5441_checked.symm
  rw [row]
  decide
theorem root466Step6 : scan edge (search edge 3) 4 [3329] 3329 root466Tail6 winner =
    scan edge (search edge 3) 4 [3329] 3329 root466Tail7 winner := by
  rw [root466Tail6, scan, if_neg (by decide), root466Candidates6_checked]
  change scan edge (search edge 3) 4 [3329] 3329 root466Tail7
    (if 3 ≤ root466Candidates6.length then search edge 3 [3329, 5441] root466Candidates6 8770 winner else winner) = _
  rw [if_neg (by decide)]
theorem root466Scan7 : scan edge (search edge 3) 4 [3329] 3329 root466Tail7 winner = winner := by rw [root466Tail7, scan, if_pos (by decide)]
theorem root466Scan6 : scan edge (search edge 3) 4 [3329] 3329 root466Tail6 winner = winner :=
  (root466Step6).trans root466Scan7
theorem root466Scan5 : scan edge (search edge 3) 4 [3329] 3329 root466Tail5 winner = winner :=
  (root466Step5).trans root466Scan6
theorem root466Scan4 : scan edge (search edge 3) 4 [3329] 3329 root466Tail4 winner = winner :=
  (root466Step4).trans root466Scan5
theorem root466Scan3 : scan edge (search edge 3) 4 [3329] 3329 root466Tail3 winner = winner :=
  (root466Step3).trans root466Scan4
theorem root466Scan2 : scan edge (search edge 3) 4 [3329] 3329 root466Tail2 winner = winner :=
  (root466Step2).trans root466Scan3
theorem root466Scan1 : scan edge (search edge 3) 4 [3329] 3329 root466Tail1 winner = winner :=
  (root466Step1).trans root466Scan2
theorem root466Scan0 : scan edge (search edge 3) 4 [3329] 3329 root466Tail0 winner = winner :=
  (root466Step0).trans root466Scan1
set_option maxRecDepth 4096 in
theorem root466Branch : search edge 4 [3329] root466Candidates 3329 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3329] 3329 root466Tail0 winner = winner
  exact root466Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
