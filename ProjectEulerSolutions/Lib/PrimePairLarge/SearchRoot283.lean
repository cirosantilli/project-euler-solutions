import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root283Candidates : List Nat := Data.row1871
set_option maxRecDepth 8192 in
theorem root283Candidates_checked : candidates edge 5 0 1871 tail284 winner.bound = root283Candidates := by
  apply candidates_eq_row edge 5 0 1871 tail284 root283Candidates winner.bound tail284_ordered
  · decide
  · decide
  · funext q
    rfl
def root283Tail48 : List Nat := []
def root283Tail47 : List Nat := 22973 :: root283Tail48
def root283Tail46 : List Nat := 22853 :: root283Tail47
def root283Tail45 : List Nat := 22511 :: root283Tail46
def root283Tail44 : List Nat := 22157 :: root283Tail45
def root283Tail43 : List Nat := 22037 :: root283Tail44
def root283Tail42 : List Nat := 21377 :: root283Tail43
def root283Tail41 : List Nat := 21149 :: root283Tail42
def root283Tail40 : List Nat := 20333 :: root283Tail41
def root283Tail39 : List Nat := 19991 :: root283Tail40
def root283Tail38 : List Nat := 19949 :: root283Tail39
def root283Tail37 : List Nat := 19853 :: root283Tail38
def root283Tail36 : List Nat := 19661 :: root283Tail37
def root283Tail35 : List Nat := 19013 :: root283Tail36
def root283Tail34 : List Nat := 18059 :: root283Tail35
def root283Tail33 : List Nat := 17921 :: root283Tail34
def root283Tail32 : List Nat := 17747 :: root283Tail33
def root283Tail31 : List Nat := 17729 :: root283Tail32
def root283Tail30 : List Nat := 17033 :: root283Tail31
def root283Tail29 : List Nat := 16931 :: root283Tail30
def root283Tail28 : List Nat := 16451 :: root283Tail29
def root283Tail27 : List Nat := 15881 :: root283Tail28
def root283Tail26 : List Nat := 15581 :: root283Tail27
def root283Tail25 : List Nat := 14747 :: root283Tail26
def root283Tail24 : List Nat := 13127 :: root283Tail25
def root283Tail23 : List Nat := 12503 :: root283Tail24
def root283Tail22 : List Nat := 12119 :: root283Tail23
def root283Tail21 : List Nat := 11927 :: root283Tail22
def root283Tail20 : List Nat := 11789 :: root283Tail21
def root283Tail19 : List Nat := 9791 :: root283Tail20
def root283Tail18 : List Nat := 9479 :: root283Tail19
def root283Tail17 : List Nat := 9209 :: root283Tail18
def root283Tail16 : List Nat := 8969 :: root283Tail17
def root283Tail15 : List Nat := 8807 :: root283Tail16
def root283Tail14 : List Nat := 8681 :: root283Tail15
def root283Tail13 : List Nat := 8171 :: root283Tail14
def root283Tail12 : List Nat := 7001 :: root283Tail13
def root283Tail11 : List Nat := 6959 :: root283Tail12
def root283Tail10 : List Nat := 6323 :: root283Tail11
def root283Tail9 : List Nat := 5441 :: root283Tail10
def root283Tail8 : List Nat := 5231 :: root283Tail9
def root283Tail7 : List Nat := 5051 :: root283Tail8
def root283Tail6 : List Nat := 4733 :: root283Tail7
def root283Tail5 : List Nat := 4691 :: root283Tail6
def root283Tail4 : List Nat := 4649 :: root283Tail5
def root283Tail3 : List Nat := 4517 :: root283Tail4
def root283Tail2 : List Nat := 4349 :: root283Tail3
def root283Tail1 : List Nat := 4139 :: root283Tail2
def root283Tail0 : List Nat := 3137 :: root283Tail1
def root283Candidates0 : List Nat := [4349]
set_option maxRecDepth 8192 in
theorem root283Candidates0_checked : candidates edge 4 1871 3137 root283Tail1 winner.bound = root283Candidates0 := by
  unfold candidates
  have row : edge 3137 = (SortedNatTree.contains FastRows.row3137) := by
    change Data.row3137.contains = (SortedNatTree.contains FastRows.row3137)
    exact FastRows.row3137_checked.symm
  rw [row]
  decide
theorem root283Step0 : scan edge (search edge 3) 4 [1871] 1871 root283Tail0 winner =
    scan edge (search edge 3) 4 [1871] 1871 root283Tail1 winner := by
  rw [root283Tail0, scan, if_neg (by decide), root283Candidates0_checked]
  change scan edge (search edge 3) 4 [1871] 1871 root283Tail1
    (if 3 ≤ root283Candidates0.length then search edge 3 [1871, 3137] root283Candidates0 5008 winner else winner) = _
  rw [if_neg (by decide)]
def root283Candidates1 : List Nat := [6959, 8969]
set_option maxRecDepth 8192 in
theorem root283Candidates1_checked : candidates edge 4 1871 4139 root283Tail2 winner.bound = root283Candidates1 := by
  unfold candidates
  have row : edge 4139 = (SortedNatTree.contains FastRows.row4139) := by
    change Data.row4139.contains = (SortedNatTree.contains FastRows.row4139)
    exact FastRows.row4139_checked.symm
  rw [row]
  decide
theorem root283Step1 : scan edge (search edge 3) 4 [1871] 1871 root283Tail1 winner =
    scan edge (search edge 3) 4 [1871] 1871 root283Tail2 winner := by
  rw [root283Tail1, scan, if_neg (by decide), root283Candidates1_checked]
  change scan edge (search edge 3) 4 [1871] 1871 root283Tail2
    (if 3 ≤ root283Candidates1.length then search edge 3 [1871, 4139] root283Candidates1 6010 winner else winner) = _
  rw [if_neg (by decide)]
def root283Candidates2 : List Nat := [5051, 12119]
set_option maxRecDepth 8192 in
theorem root283Candidates2_checked : candidates edge 4 1871 4349 root283Tail3 winner.bound = root283Candidates2 := by
  unfold candidates
  have row : edge 4349 = (SortedNatTree.contains FastRows.row4349) := by
    change Data.row4349.contains = (SortedNatTree.contains FastRows.row4349)
    exact FastRows.row4349_checked.symm
  rw [row]
  decide
theorem root283Step2 : scan edge (search edge 3) 4 [1871] 1871 root283Tail2 winner =
    scan edge (search edge 3) 4 [1871] 1871 root283Tail3 winner := by
  rw [root283Tail2, scan, if_neg (by decide), root283Candidates2_checked]
  change scan edge (search edge 3) 4 [1871] 1871 root283Tail3
    (if 3 ≤ root283Candidates2.length then search edge 3 [1871, 4349] root283Candidates2 6220 winner else winner) = _
  rw [if_neg (by decide)]
def root283Candidates3 : List Nat := [6959, 17729]
set_option maxRecDepth 8192 in
theorem root283Candidates3_checked : candidates edge 4 1871 4517 root283Tail4 winner.bound = root283Candidates3 := by
  unfold candidates
  have row : edge 4517 = (SortedNatTree.contains FastRows.row4517) := by
    change Data.row4517.contains = (SortedNatTree.contains FastRows.row4517)
    exact FastRows.row4517_checked.symm
  rw [row]
  decide
theorem root283Step3 : scan edge (search edge 3) 4 [1871] 1871 root283Tail3 winner =
    scan edge (search edge 3) 4 [1871] 1871 root283Tail4 winner := by
  rw [root283Tail3, scan, if_neg (by decide), root283Candidates3_checked]
  change scan edge (search edge 3) 4 [1871] 1871 root283Tail4
    (if 3 ≤ root283Candidates3.length then search edge 3 [1871, 4517] root283Candidates3 6388 winner else winner) = _
  rw [if_neg (by decide)]
def root283Candidates4 : List Nat := [6323, 18059]
set_option maxRecDepth 8192 in
theorem root283Candidates4_checked : candidates edge 4 1871 4649 root283Tail5 winner.bound = root283Candidates4 := by
  unfold candidates
  have row : edge 4649 = (SortedNatTree.contains FastRows.row4649) := by
    change Data.row4649.contains = (SortedNatTree.contains FastRows.row4649)
    exact FastRows.row4649_checked.symm
  rw [row]
  decide
theorem root283Step4 : scan edge (search edge 3) 4 [1871] 1871 root283Tail4 winner =
    scan edge (search edge 3) 4 [1871] 1871 root283Tail5 winner := by
  rw [root283Tail4, scan, if_neg (by decide), root283Candidates4_checked]
  change scan edge (search edge 3) 4 [1871] 1871 root283Tail5
    (if 3 ≤ root283Candidates4.length then search edge 3 [1871, 4649] root283Candidates4 6520 winner else winner) = _
  rw [if_neg (by decide)]
def root283Candidates5 : List Nat := [17921]
set_option maxRecDepth 8192 in
theorem root283Candidates5_checked : candidates edge 4 1871 4691 root283Tail6 winner.bound = root283Candidates5 := by
  unfold candidates
  have row : edge 4691 = (SortedNatTree.contains FastRows.row4691) := by
    change Data.row4691.contains = (SortedNatTree.contains FastRows.row4691)
    exact FastRows.row4691_checked.symm
  rw [row]
  decide
theorem root283Step5 : scan edge (search edge 3) 4 [1871] 1871 root283Tail5 winner =
    scan edge (search edge 3) 4 [1871] 1871 root283Tail6 winner := by
  rw [root283Tail5, scan, if_neg (by decide), root283Candidates5_checked]
  change scan edge (search edge 3) 4 [1871] 1871 root283Tail6
    (if 3 ≤ root283Candidates5.length then search edge 3 [1871, 4691] root283Candidates5 6562 winner else winner) = _
  rw [if_neg (by decide)]
def root283Candidates6 : List Nat := []
set_option maxRecDepth 8192 in
theorem root283Candidates6_checked : candidates edge 4 1871 4733 root283Tail7 winner.bound = root283Candidates6 := by
  unfold candidates
  have row : edge 4733 = (SortedNatTree.contains FastRows.row4733) := by
    change Data.row4733.contains = (SortedNatTree.contains FastRows.row4733)
    exact FastRows.row4733_checked.symm
  rw [row]
  decide
theorem root283Step6 : scan edge (search edge 3) 4 [1871] 1871 root283Tail6 winner =
    scan edge (search edge 3) 4 [1871] 1871 root283Tail7 winner := by
  rw [root283Tail6, scan, if_neg (by decide), root283Candidates6_checked]
  change scan edge (search edge 3) 4 [1871] 1871 root283Tail7
    (if 3 ≤ root283Candidates6.length then search edge 3 [1871, 4733] root283Candidates6 6604 winner else winner) = _
  rw [if_neg (by decide)]
def root283Candidates7 : List Nat := [8969, 17729]
set_option maxRecDepth 8192 in
theorem root283Candidates7_checked : candidates edge 4 1871 5051 root283Tail8 winner.bound = root283Candidates7 := by
  unfold candidates
  have row : edge 5051 = (SortedNatTree.contains FastRows.row5051) := by
    change Data.row5051.contains = (SortedNatTree.contains FastRows.row5051)
    exact FastRows.row5051_checked.symm
  rw [row]
  decide
theorem root283Step7 : scan edge (search edge 3) 4 [1871] 1871 root283Tail7 winner =
    scan edge (search edge 3) 4 [1871] 1871 root283Tail8 winner := by
  rw [root283Tail7, scan, if_neg (by decide), root283Candidates7_checked]
  change scan edge (search edge 3) 4 [1871] 1871 root283Tail8
    (if 3 ≤ root283Candidates7.length then search edge 3 [1871, 5051] root283Candidates7 6922 winner else winner) = _
  rw [if_neg (by decide)]
def root283Candidates8 : List Nat := []
set_option maxRecDepth 8192 in
theorem root283Candidates8_checked : candidates edge 4 1871 5231 root283Tail9 winner.bound = root283Candidates8 := by
  unfold candidates
  have row : edge 5231 = (SortedNatTree.contains FastRows.row5231) := by
    change Data.row5231.contains = (SortedNatTree.contains FastRows.row5231)
    exact FastRows.row5231_checked.symm
  rw [row]
  decide
theorem root283Step8 : scan edge (search edge 3) 4 [1871] 1871 root283Tail8 winner =
    scan edge (search edge 3) 4 [1871] 1871 root283Tail9 winner := by
  rw [root283Tail8, scan, if_neg (by decide), root283Candidates8_checked]
  change scan edge (search edge 3) 4 [1871] 1871 root283Tail9
    (if 3 ≤ root283Candidates8.length then search edge 3 [1871, 5231] root283Candidates8 7102 winner else winner) = _
  rw [if_neg (by decide)]
def root283Candidates9 : List Nat := [6959, 11927]
set_option maxRecDepth 8192 in
theorem root283Candidates9_checked : candidates edge 4 1871 5441 root283Tail10 winner.bound = root283Candidates9 := by
  unfold candidates
  have row : edge 5441 = (SortedNatTree.contains FastRows.row5441) := by
    change Data.row5441.contains = (SortedNatTree.contains FastRows.row5441)
    exact FastRows.row5441_checked.symm
  rw [row]
  decide
theorem root283Step9 : scan edge (search edge 3) 4 [1871] 1871 root283Tail9 winner =
    scan edge (search edge 3) 4 [1871] 1871 root283Tail10 winner := by
  rw [root283Tail9, scan, if_neg (by decide), root283Candidates9_checked]
  change scan edge (search edge 3) 4 [1871] 1871 root283Tail10
    (if 3 ≤ root283Candidates9.length then search edge 3 [1871, 5441] root283Candidates9 7312 winner else winner) = _
  rw [if_neg (by decide)]
theorem root283Scan10 : scan edge (search edge 3) 4 [1871] 1871 root283Tail10 winner = winner := by rw [root283Tail10, scan, if_pos (by decide)]
theorem root283Scan9 : scan edge (search edge 3) 4 [1871] 1871 root283Tail9 winner = winner :=
  (root283Step9).trans root283Scan10
theorem root283Scan8 : scan edge (search edge 3) 4 [1871] 1871 root283Tail8 winner = winner :=
  (root283Step8).trans root283Scan9
theorem root283Scan7 : scan edge (search edge 3) 4 [1871] 1871 root283Tail7 winner = winner :=
  (root283Step7).trans root283Scan8
theorem root283Scan6 : scan edge (search edge 3) 4 [1871] 1871 root283Tail6 winner = winner :=
  (root283Step6).trans root283Scan7
theorem root283Scan5 : scan edge (search edge 3) 4 [1871] 1871 root283Tail5 winner = winner :=
  (root283Step5).trans root283Scan6
theorem root283Scan4 : scan edge (search edge 3) 4 [1871] 1871 root283Tail4 winner = winner :=
  (root283Step4).trans root283Scan5
theorem root283Scan3 : scan edge (search edge 3) 4 [1871] 1871 root283Tail3 winner = winner :=
  (root283Step3).trans root283Scan4
theorem root283Scan2 : scan edge (search edge 3) 4 [1871] 1871 root283Tail2 winner = winner :=
  (root283Step2).trans root283Scan3
theorem root283Scan1 : scan edge (search edge 3) 4 [1871] 1871 root283Tail1 winner = winner :=
  (root283Step1).trans root283Scan2
theorem root283Scan0 : scan edge (search edge 3) 4 [1871] 1871 root283Tail0 winner = winner :=
  (root283Step0).trans root283Scan1
set_option maxRecDepth 4096 in
theorem root283Branch : search edge 4 [1871] root283Candidates 1871 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1871] 1871 root283Tail0 winner = winner
  exact root283Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
