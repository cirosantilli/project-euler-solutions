import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root335Candidates : List Nat := Data.row2273
set_option maxRecDepth 8192 in
theorem root335Candidates_checked : candidates edge 5 0 2273 tail336 winner.bound = root335Candidates := by
  apply candidates_eq_row edge 5 0 2273 tail336 root335Candidates winner.bound tail336_ordered
  · decide
  · decide
  · funext q
    rfl
def root335Tail48 : List Nat := []
def root335Tail47 : List Nat := 23333 :: root335Tail48
def root335Tail46 : List Nat := 22259 :: root335Tail47
def root335Tail45 : List Nat := 21911 :: root335Tail46
def root335Tail44 : List Nat := 21023 :: root335Tail45
def root335Tail43 : List Nat := 20963 :: root335Tail44
def root335Tail42 : List Nat := 20543 :: root335Tail43
def root335Tail41 : List Nat := 20147 :: root335Tail42
def root335Tail40 : List Nat := 19997 :: root335Tail41
def root335Tail39 : List Nat := 19181 :: root335Tail40
def root335Tail38 : List Nat := 19013 :: root335Tail39
def root335Tail37 : List Nat := 18797 :: root335Tail38
def root335Tail36 : List Nat := 18353 :: root335Tail37
def root335Tail35 : List Nat := 16787 :: root335Tail36
def root335Tail34 : List Nat := 16553 :: root335Tail35
def root335Tail33 : List Nat := 16253 :: root335Tail34
def root335Tail32 : List Nat := 15017 :: root335Tail33
def root335Tail31 : List Nat := 14387 :: root335Tail32
def root335Tail30 : List Nat := 14369 :: root335Tail31
def root335Tail29 : List Nat := 14057 :: root335Tail30
def root335Tail28 : List Nat := 12809 :: root335Tail29
def root335Tail27 : List Nat := 12689 :: root335Tail28
def root335Tail26 : List Nat := 12227 :: root335Tail27
def root335Tail25 : List Nat := 11597 :: root335Tail26
def root335Tail24 : List Nat := 11321 :: root335Tail25
def root335Tail23 : List Nat := 11087 :: root335Tail24
def root335Tail22 : List Nat := 11027 :: root335Tail23
def root335Tail21 : List Nat := 10739 :: root335Tail22
def root335Tail20 : List Nat := 10037 :: root335Tail21
def root335Tail19 : List Nat := 9791 :: root335Tail20
def root335Tail18 : List Nat := 9323 :: root335Tail19
def root335Tail17 : List Nat := 8861 :: root335Tail18
def root335Tail16 : List Nat := 8669 :: root335Tail17
def root335Tail15 : List Nat := 8627 :: root335Tail16
def root335Tail14 : List Nat := 8297 :: root335Tail15
def root335Tail13 : List Nat := 8147 :: root335Tail14
def root335Tail12 : List Nat := 7283 :: root335Tail13
def root335Tail11 : List Nat := 5897 :: root335Tail12
def root335Tail10 : List Nat := 5861 :: root335Tail11
def root335Tail9 : List Nat := 5711 :: root335Tail10
def root335Tail8 : List Nat := 5399 :: root335Tail9
def root335Tail7 : List Nat := 4973 :: root335Tail8
def root335Tail6 : List Nat := 3761 :: root335Tail7
def root335Tail5 : List Nat := 3461 :: root335Tail6
def root335Tail4 : List Nat := 3041 :: root335Tail5
def root335Tail3 : List Nat := 2999 :: root335Tail4
def root335Tail2 : List Nat := 2903 :: root335Tail3
def root335Tail1 : List Nat := 2789 :: root335Tail2
def root335Tail0 : List Nat := 2609 :: root335Tail1
def root335Candidates0 : List Nat := [4973, 8297, 10037, 14387]
set_option maxRecDepth 8192 in
theorem root335Candidates0_checked : candidates edge 4 2273 2609 root335Tail1 winner.bound = root335Candidates0 := by
  unfold candidates
  have row : edge 2609 = (SortedNatTree.contains FastRows.row2609) := by
    change Data.row2609.contains = (SortedNatTree.contains FastRows.row2609)
    exact FastRows.row2609_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root335Child0 : search edge 3 [2273, 2609] root335Candidates0 4882 winner = winner := by decide
theorem root335Step0 : scan edge (search edge 3) 4 [2273] 2273 root335Tail0 winner =
    scan edge (search edge 3) 4 [2273] 2273 root335Tail1 winner := by
  rw [root335Tail0, scan, if_neg (by decide), root335Candidates0_checked]
  change scan edge (search edge 3) 4 [2273] 2273 root335Tail1
    (if 3 ≤ root335Candidates0.length then search edge 3 [2273, 2609] root335Candidates0 4882 winner else winner) = _
  rw [if_pos (by decide), root335Child0]
def root335Candidates1 : List Nat := [3461]
set_option maxRecDepth 8192 in
theorem root335Candidates1_checked : candidates edge 4 2273 2789 root335Tail2 winner.bound = root335Candidates1 := by
  unfold candidates
  have row : edge 2789 = (SortedNatTree.contains FastRows.row2789) := by
    change Data.row2789.contains = (SortedNatTree.contains FastRows.row2789)
    exact FastRows.row2789_checked.symm
  rw [row]
  decide
theorem root335Step1 : scan edge (search edge 3) 4 [2273] 2273 root335Tail1 winner =
    scan edge (search edge 3) 4 [2273] 2273 root335Tail2 winner := by
  rw [root335Tail1, scan, if_neg (by decide), root335Candidates1_checked]
  change scan edge (search edge 3) 4 [2273] 2273 root335Tail2
    (if 3 ≤ root335Candidates1.length then search edge 3 [2273, 2789] root335Candidates1 5062 winner else winner) = _
  rw [if_neg (by decide)]
def root335Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root335Candidates2_checked : candidates edge 4 2273 2903 root335Tail3 winner.bound = root335Candidates2 := by
  unfold candidates
  have row : edge 2903 = (SortedNatTree.contains FastRows.row2903) := by
    change Data.row2903.contains = (SortedNatTree.contains FastRows.row2903)
    exact FastRows.row2903_checked.symm
  rw [row]
  decide
theorem root335Step2 : scan edge (search edge 3) 4 [2273] 2273 root335Tail2 winner =
    scan edge (search edge 3) 4 [2273] 2273 root335Tail3 winner := by
  rw [root335Tail2, scan, if_neg (by decide), root335Candidates2_checked]
  change scan edge (search edge 3) 4 [2273] 2273 root335Tail3
    (if 3 ≤ root335Candidates2.length then search edge 3 [2273, 2903] root335Candidates2 5176 winner else winner) = _
  rw [if_neg (by decide)]
def root335Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root335Candidates3_checked : candidates edge 4 2273 2999 root335Tail4 winner.bound = root335Candidates3 := by
  unfold candidates
  have row : edge 2999 = (SortedNatTree.contains FastRows.row2999) := by
    change Data.row2999.contains = (SortedNatTree.contains FastRows.row2999)
    exact FastRows.row2999_checked.symm
  rw [row]
  decide
theorem root335Step3 : scan edge (search edge 3) 4 [2273] 2273 root335Tail3 winner =
    scan edge (search edge 3) 4 [2273] 2273 root335Tail4 winner := by
  rw [root335Tail3, scan, if_neg (by decide), root335Candidates3_checked]
  change scan edge (search edge 3) 4 [2273] 2273 root335Tail4
    (if 3 ≤ root335Candidates3.length then search edge 3 [2273, 2999] root335Candidates3 5272 winner else winner) = _
  rw [if_neg (by decide)]
def root335Candidates4 : List Nat := [5897]
set_option maxRecDepth 8192 in
theorem root335Candidates4_checked : candidates edge 4 2273 3041 root335Tail5 winner.bound = root335Candidates4 := by
  unfold candidates
  have row : edge 3041 = (SortedNatTree.contains FastRows.row3041) := by
    change Data.row3041.contains = (SortedNatTree.contains FastRows.row3041)
    exact FastRows.row3041_checked.symm
  rw [row]
  decide
theorem root335Step4 : scan edge (search edge 3) 4 [2273] 2273 root335Tail4 winner =
    scan edge (search edge 3) 4 [2273] 2273 root335Tail5 winner := by
  rw [root335Tail4, scan, if_neg (by decide), root335Candidates4_checked]
  change scan edge (search edge 3) 4 [2273] 2273 root335Tail5
    (if 3 ≤ root335Candidates4.length then search edge 3 [2273, 3041] root335Candidates4 5314 winner else winner) = _
  rw [if_neg (by decide)]
def root335Candidates5 : List Nat := []
set_option maxRecDepth 8192 in
theorem root335Candidates5_checked : candidates edge 4 2273 3461 root335Tail6 winner.bound = root335Candidates5 := by
  unfold candidates
  have row : edge 3461 = (SortedNatTree.contains FastRows.row3461) := by
    change Data.row3461.contains = (SortedNatTree.contains FastRows.row3461)
    exact FastRows.row3461_checked.symm
  rw [row]
  decide
theorem root335Step5 : scan edge (search edge 3) 4 [2273] 2273 root335Tail5 winner =
    scan edge (search edge 3) 4 [2273] 2273 root335Tail6 winner := by
  rw [root335Tail5, scan, if_neg (by decide), root335Candidates5_checked]
  change scan edge (search edge 3) 4 [2273] 2273 root335Tail6
    (if 3 ≤ root335Candidates5.length then search edge 3 [2273, 3461] root335Candidates5 5734 winner else winner) = _
  rw [if_neg (by decide)]
def root335Candidates6 : List Nat := [10037, 16553]
set_option maxRecDepth 8192 in
theorem root335Candidates6_checked : candidates edge 4 2273 3761 root335Tail7 winner.bound = root335Candidates6 := by
  unfold candidates
  have row : edge 3761 = (SortedNatTree.contains FastRows.row3761) := by
    change Data.row3761.contains = (SortedNatTree.contains FastRows.row3761)
    exact FastRows.row3761_checked.symm
  rw [row]
  decide
theorem root335Step6 : scan edge (search edge 3) 4 [2273] 2273 root335Tail6 winner =
    scan edge (search edge 3) 4 [2273] 2273 root335Tail7 winner := by
  rw [root335Tail6, scan, if_neg (by decide), root335Candidates6_checked]
  change scan edge (search edge 3) 4 [2273] 2273 root335Tail7
    (if 3 ≤ root335Candidates6.length then search edge 3 [2273, 3761] root335Candidates6 6034 winner else winner) = _
  rw [if_neg (by decide)]
def root335Candidates7 : List Nat := []
set_option maxRecDepth 8192 in
theorem root335Candidates7_checked : candidates edge 4 2273 4973 root335Tail8 winner.bound = root335Candidates7 := by
  unfold candidates
  have row : edge 4973 = (SortedNatTree.contains FastRows.row4973) := by
    change Data.row4973.contains = (SortedNatTree.contains FastRows.row4973)
    exact FastRows.row4973_checked.symm
  rw [row]
  decide
theorem root335Step7 : scan edge (search edge 3) 4 [2273] 2273 root335Tail7 winner =
    scan edge (search edge 3) 4 [2273] 2273 root335Tail8 winner := by
  rw [root335Tail7, scan, if_neg (by decide), root335Candidates7_checked]
  change scan edge (search edge 3) 4 [2273] 2273 root335Tail8
    (if 3 ≤ root335Candidates7.length then search edge 3 [2273, 4973] root335Candidates7 7246 winner else winner) = _
  rw [if_neg (by decide)]
def root335Candidates8 : List Nat := [10037]
set_option maxRecDepth 8192 in
theorem root335Candidates8_checked : candidates edge 4 2273 5399 root335Tail9 winner.bound = root335Candidates8 := by
  unfold candidates
  have row : edge 5399 = (SortedNatTree.contains FastRows.row5399) := by
    change Data.row5399.contains = (SortedNatTree.contains FastRows.row5399)
    exact FastRows.row5399_checked.symm
  rw [row]
  decide
theorem root335Step8 : scan edge (search edge 3) 4 [2273] 2273 root335Tail8 winner =
    scan edge (search edge 3) 4 [2273] 2273 root335Tail9 winner := by
  rw [root335Tail8, scan, if_neg (by decide), root335Candidates8_checked]
  change scan edge (search edge 3) 4 [2273] 2273 root335Tail9
    (if 3 ≤ root335Candidates8.length then search edge 3 [2273, 5399] root335Candidates8 7672 winner else winner) = _
  rw [if_neg (by decide)]
def root335Candidates9 : List Nat := []
set_option maxRecDepth 8192 in
theorem root335Candidates9_checked : candidates edge 4 2273 5711 root335Tail10 winner.bound = root335Candidates9 := by
  unfold candidates
  have row : edge 5711 = (SortedNatTree.contains FastRows.row5711) := by
    change Data.row5711.contains = (SortedNatTree.contains FastRows.row5711)
    exact FastRows.row5711_checked.symm
  rw [row]
  decide
theorem root335Step9 : scan edge (search edge 3) 4 [2273] 2273 root335Tail9 winner =
    scan edge (search edge 3) 4 [2273] 2273 root335Tail10 winner := by
  rw [root335Tail9, scan, if_neg (by decide), root335Candidates9_checked]
  change scan edge (search edge 3) 4 [2273] 2273 root335Tail10
    (if 3 ≤ root335Candidates9.length then search edge 3 [2273, 5711] root335Candidates9 7984 winner else winner) = _
  rw [if_neg (by decide)]
def root335Candidates10 : List Nat := []
set_option maxRecDepth 8192 in
theorem root335Candidates10_checked : candidates edge 4 2273 5861 root335Tail11 winner.bound = root335Candidates10 := by
  unfold candidates
  have row : edge 5861 = (SortedNatTree.contains FastRows.row5861) := by
    change Data.row5861.contains = (SortedNatTree.contains FastRows.row5861)
    exact FastRows.row5861_checked.symm
  rw [row]
  decide
theorem root335Step10 : scan edge (search edge 3) 4 [2273] 2273 root335Tail10 winner =
    scan edge (search edge 3) 4 [2273] 2273 root335Tail11 winner := by
  rw [root335Tail10, scan, if_neg (by decide), root335Candidates10_checked]
  change scan edge (search edge 3) 4 [2273] 2273 root335Tail11
    (if 3 ≤ root335Candidates10.length then search edge 3 [2273, 5861] root335Candidates10 8134 winner else winner) = _
  rw [if_neg (by decide)]
def root335Candidates11 : List Nat := [8861]
set_option maxRecDepth 8192 in
theorem root335Candidates11_checked : candidates edge 4 2273 5897 root335Tail12 winner.bound = root335Candidates11 := by
  unfold candidates
  have row : edge 5897 = (SortedNatTree.contains FastRows.row5897) := by
    change Data.row5897.contains = (SortedNatTree.contains FastRows.row5897)
    exact FastRows.row5897_checked.symm
  rw [row]
  decide
theorem root335Step11 : scan edge (search edge 3) 4 [2273] 2273 root335Tail11 winner =
    scan edge (search edge 3) 4 [2273] 2273 root335Tail12 winner := by
  rw [root335Tail11, scan, if_neg (by decide), root335Candidates11_checked]
  change scan edge (search edge 3) 4 [2273] 2273 root335Tail12
    (if 3 ≤ root335Candidates11.length then search edge 3 [2273, 5897] root335Candidates11 8170 winner else winner) = _
  rw [if_neg (by decide)]
theorem root335Scan12 : scan edge (search edge 3) 4 [2273] 2273 root335Tail12 winner = winner := by rw [root335Tail12, scan, if_pos (by decide)]
theorem root335Scan11 : scan edge (search edge 3) 4 [2273] 2273 root335Tail11 winner = winner :=
  (root335Step11).trans root335Scan12
theorem root335Scan10 : scan edge (search edge 3) 4 [2273] 2273 root335Tail10 winner = winner :=
  (root335Step10).trans root335Scan11
theorem root335Scan9 : scan edge (search edge 3) 4 [2273] 2273 root335Tail9 winner = winner :=
  (root335Step9).trans root335Scan10
theorem root335Scan8 : scan edge (search edge 3) 4 [2273] 2273 root335Tail8 winner = winner :=
  (root335Step8).trans root335Scan9
theorem root335Scan7 : scan edge (search edge 3) 4 [2273] 2273 root335Tail7 winner = winner :=
  (root335Step7).trans root335Scan8
theorem root335Scan6 : scan edge (search edge 3) 4 [2273] 2273 root335Tail6 winner = winner :=
  (root335Step6).trans root335Scan7
theorem root335Scan5 : scan edge (search edge 3) 4 [2273] 2273 root335Tail5 winner = winner :=
  (root335Step5).trans root335Scan6
theorem root335Scan4 : scan edge (search edge 3) 4 [2273] 2273 root335Tail4 winner = winner :=
  (root335Step4).trans root335Scan5
theorem root335Scan3 : scan edge (search edge 3) 4 [2273] 2273 root335Tail3 winner = winner :=
  (root335Step3).trans root335Scan4
theorem root335Scan2 : scan edge (search edge 3) 4 [2273] 2273 root335Tail2 winner = winner :=
  (root335Step2).trans root335Scan3
theorem root335Scan1 : scan edge (search edge 3) 4 [2273] 2273 root335Tail1 winner = winner :=
  (root335Step1).trans root335Scan2
theorem root335Scan0 : scan edge (search edge 3) 4 [2273] 2273 root335Tail0 winner = winner :=
  (root335Step0).trans root335Scan1
set_option maxRecDepth 4096 in
theorem root335Branch : search edge 4 [2273] root335Candidates 2273 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2273] 2273 root335Tail0 winner = winner
  exact root335Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
