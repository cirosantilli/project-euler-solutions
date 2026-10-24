import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root457Candidates : List Nat := Data.row3257
set_option maxRecDepth 8192 in
theorem root457Candidates_checked : candidates edge 5 0 3257 tail458 winner.bound = root457Candidates := by
  apply candidates_eq_row edge 5 0 3257 tail458 root457Candidates winner.bound tail458_ordered
  · decide
  · decide
  · funext q
    rfl
def root457Tail36 : List Nat := []
def root457Tail35 : List Nat := 21191 :: root457Tail36
def root457Tail34 : List Nat := 20939 :: root457Tail35
def root457Tail33 : List Nat := 20333 :: root457Tail34
def root457Tail32 : List Nat := 18713 :: root457Tail33
def root457Tail31 : List Nat := 18257 :: root457Tail32
def root457Tail30 : List Nat := 18191 :: root457Tail31
def root457Tail29 : List Nat := 18089 :: root457Tail30
def root457Tail28 : List Nat := 17519 :: root457Tail29
def root457Tail27 : List Nat := 17417 :: root457Tail28
def root457Tail26 : List Nat := 13679 :: root457Tail27
def root457Tail25 : List Nat := 13151 :: root457Tail26
def root457Tail24 : List Nat := 13127 :: root457Tail25
def root457Tail23 : List Nat := 12923 :: root457Tail24
def root457Tail22 : List Nat := 12347 :: root457Tail23
def root457Tail21 : List Nat := 12251 :: root457Tail22
def root457Tail20 : List Nat := 11807 :: root457Tail21
def root457Tail19 : List Nat := 11261 :: root457Tail20
def root457Tail18 : List Nat := 11069 :: root457Tail19
def root457Tail17 : List Nat := 11003 :: root457Tail18
def root457Tail16 : List Nat := 10709 :: root457Tail17
def root457Tail15 : List Nat := 9587 :: root457Tail16
def root457Tail14 : List Nat := 9059 :: root457Tail15
def root457Tail13 : List Nat := 7949 :: root457Tail14
def root457Tail12 : List Nat := 6917 :: root457Tail13
def root457Tail11 : List Nat := 6491 :: root457Tail12
def root457Tail10 : List Nat := 6221 :: root457Tail11
def root457Tail9 : List Nat := 6053 :: root457Tail10
def root457Tail8 : List Nat := 5507 :: root457Tail9
def root457Tail7 : List Nat := 5471 :: root457Tail8
def root457Tail6 : List Nat := 5441 :: root457Tail7
def root457Tail5 : List Nat := 5051 :: root457Tail6
def root457Tail4 : List Nat := 4937 :: root457Tail5
def root457Tail3 : List Nat := 4649 :: root457Tail4
def root457Tail2 : List Nat := 4547 :: root457Tail3
def root457Tail1 : List Nat := 3449 :: root457Tail2
def root457Tail0 : List Nat := 3407 :: root457Tail1
def root457Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root457Candidates0_checked : candidates edge 4 3257 3407 root457Tail1 winner.bound = root457Candidates0 := by
  unfold candidates
  have row : edge 3407 = (SortedNatTree.contains FastRows.row3407) := by
    change Data.row3407.contains = (SortedNatTree.contains FastRows.row3407)
    exact FastRows.row3407_checked.symm
  rw [row]
  decide
theorem root457Step0 : scan edge (search edge 3) 4 [3257] 3257 root457Tail0 winner =
    scan edge (search edge 3) 4 [3257] 3257 root457Tail1 winner := by
  rw [root457Tail0, scan, if_neg (by decide), root457Candidates0_checked]
  change scan edge (search edge 3) 4 [3257] 3257 root457Tail1
    (if 3 ≤ root457Candidates0.length then search edge 3 [3257, 3407] root457Candidates0 6664 winner else winner) = _
  rw [if_neg (by decide)]
def root457Candidates1 : List Nat := [18713]
set_option maxRecDepth 8192 in
theorem root457Candidates1_checked : candidates edge 4 3257 3449 root457Tail2 winner.bound = root457Candidates1 := by
  unfold candidates
  have row : edge 3449 = (SortedNatTree.contains FastRows.row3449) := by
    change Data.row3449.contains = (SortedNatTree.contains FastRows.row3449)
    exact FastRows.row3449_checked.symm
  rw [row]
  decide
theorem root457Step1 : scan edge (search edge 3) 4 [3257] 3257 root457Tail1 winner =
    scan edge (search edge 3) 4 [3257] 3257 root457Tail2 winner := by
  rw [root457Tail1, scan, if_neg (by decide), root457Candidates1_checked]
  change scan edge (search edge 3) 4 [3257] 3257 root457Tail2
    (if 3 ≤ root457Candidates1.length then search edge 3 [3257, 3449] root457Candidates1 6706 winner else winner) = _
  rw [if_neg (by decide)]
def root457Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root457Candidates2_checked : candidates edge 4 3257 4547 root457Tail3 winner.bound = root457Candidates2 := by
  unfold candidates
  have row : edge 4547 = (SortedNatTree.contains FastRows.row4547) := by
    change Data.row4547.contains = (SortedNatTree.contains FastRows.row4547)
    exact FastRows.row4547_checked.symm
  rw [row]
  decide
theorem root457Step2 : scan edge (search edge 3) 4 [3257] 3257 root457Tail2 winner =
    scan edge (search edge 3) 4 [3257] 3257 root457Tail3 winner := by
  rw [root457Tail2, scan, if_neg (by decide), root457Candidates2_checked]
  change scan edge (search edge 3) 4 [3257] 3257 root457Tail3
    (if 3 ≤ root457Candidates2.length then search edge 3 [3257, 4547] root457Candidates2 7804 winner else winner) = _
  rw [if_neg (by decide)]
def root457Candidates3 : List Nat := [12251]
set_option maxRecDepth 8192 in
theorem root457Candidates3_checked : candidates edge 4 3257 4649 root457Tail4 winner.bound = root457Candidates3 := by
  unfold candidates
  have row : edge 4649 = (SortedNatTree.contains FastRows.row4649) := by
    change Data.row4649.contains = (SortedNatTree.contains FastRows.row4649)
    exact FastRows.row4649_checked.symm
  rw [row]
  decide
theorem root457Step3 : scan edge (search edge 3) 4 [3257] 3257 root457Tail3 winner =
    scan edge (search edge 3) 4 [3257] 3257 root457Tail4 winner := by
  rw [root457Tail3, scan, if_neg (by decide), root457Candidates3_checked]
  change scan edge (search edge 3) 4 [3257] 3257 root457Tail4
    (if 3 ≤ root457Candidates3.length then search edge 3 [3257, 4649] root457Candidates3 7906 winner else winner) = _
  rw [if_neg (by decide)]
def root457Candidates4 : List Nat := [6053, 6491]
set_option maxRecDepth 8192 in
theorem root457Candidates4_checked : candidates edge 4 3257 4937 root457Tail5 winner.bound = root457Candidates4 := by
  unfold candidates
  have row : edge 4937 = (SortedNatTree.contains FastRows.row4937) := by
    change Data.row4937.contains = (SortedNatTree.contains FastRows.row4937)
    exact FastRows.row4937_checked.symm
  rw [row]
  decide
theorem root457Step4 : scan edge (search edge 3) 4 [3257] 3257 root457Tail4 winner =
    scan edge (search edge 3) 4 [3257] 3257 root457Tail5 winner := by
  rw [root457Tail4, scan, if_neg (by decide), root457Candidates4_checked]
  change scan edge (search edge 3) 4 [3257] 3257 root457Tail5
    (if 3 ≤ root457Candidates4.length then search edge 3 [3257, 4937] root457Candidates4 8194 winner else winner) = _
  rw [if_neg (by decide)]
def root457Candidates5 : List Nat := [7949]
set_option maxRecDepth 8192 in
theorem root457Candidates5_checked : candidates edge 4 3257 5051 root457Tail6 winner.bound = root457Candidates5 := by
  unfold candidates
  have row : edge 5051 = (SortedNatTree.contains FastRows.row5051) := by
    change Data.row5051.contains = (SortedNatTree.contains FastRows.row5051)
    exact FastRows.row5051_checked.symm
  rw [row]
  decide
theorem root457Step5 : scan edge (search edge 3) 4 [3257] 3257 root457Tail5 winner =
    scan edge (search edge 3) 4 [3257] 3257 root457Tail6 winner := by
  rw [root457Tail5, scan, if_neg (by decide), root457Candidates5_checked]
  change scan edge (search edge 3) 4 [3257] 3257 root457Tail6
    (if 3 ≤ root457Candidates5.length then search edge 3 [3257, 5051] root457Candidates5 8308 winner else winner) = _
  rw [if_neg (by decide)]
def root457Candidates6 : List Nat := [13151]
set_option maxRecDepth 8192 in
theorem root457Candidates6_checked : candidates edge 4 3257 5441 root457Tail7 winner.bound = root457Candidates6 := by
  unfold candidates
  have row : edge 5441 = (SortedNatTree.contains FastRows.row5441) := by
    change Data.row5441.contains = (SortedNatTree.contains FastRows.row5441)
    exact FastRows.row5441_checked.symm
  rw [row]
  decide
theorem root457Step6 : scan edge (search edge 3) 4 [3257] 3257 root457Tail6 winner =
    scan edge (search edge 3) 4 [3257] 3257 root457Tail7 winner := by
  rw [root457Tail6, scan, if_neg (by decide), root457Candidates6_checked]
  change scan edge (search edge 3) 4 [3257] 3257 root457Tail7
    (if 3 ≤ root457Candidates6.length then search edge 3 [3257, 5441] root457Candidates6 8698 winner else winner) = _
  rw [if_neg (by decide)]
def root457Candidates7 : List Nat := [6491]
set_option maxRecDepth 8192 in
theorem root457Candidates7_checked : candidates edge 4 3257 5471 root457Tail8 winner.bound = root457Candidates7 := by
  unfold candidates
  have row : edge 5471 = (SortedNatTree.contains FastRows.row5471) := by
    change Data.row5471.contains = (SortedNatTree.contains FastRows.row5471)
    exact FastRows.row5471_checked.symm
  rw [row]
  decide
theorem root457Step7 : scan edge (search edge 3) 4 [3257] 3257 root457Tail7 winner =
    scan edge (search edge 3) 4 [3257] 3257 root457Tail8 winner := by
  rw [root457Tail7, scan, if_neg (by decide), root457Candidates7_checked]
  change scan edge (search edge 3) 4 [3257] 3257 root457Tail8
    (if 3 ≤ root457Candidates7.length then search edge 3 [3257, 5471] root457Candidates7 8728 winner else winner) = _
  rw [if_neg (by decide)]
def root457Candidates8 : List Nat := [6221]
set_option maxRecDepth 8192 in
theorem root457Candidates8_checked : candidates edge 4 3257 5507 root457Tail9 winner.bound = root457Candidates8 := by
  unfold candidates
  have row : edge 5507 = (SortedNatTree.contains FastRows.row5507) := by
    change Data.row5507.contains = (SortedNatTree.contains FastRows.row5507)
    exact FastRows.row5507_checked.symm
  rw [row]
  decide
theorem root457Step8 : scan edge (search edge 3) 4 [3257] 3257 root457Tail8 winner =
    scan edge (search edge 3) 4 [3257] 3257 root457Tail9 winner := by
  rw [root457Tail8, scan, if_neg (by decide), root457Candidates8_checked]
  change scan edge (search edge 3) 4 [3257] 3257 root457Tail9
    (if 3 ≤ root457Candidates8.length then search edge 3 [3257, 5507] root457Candidates8 8764 winner else winner) = _
  rw [if_neg (by decide)]
theorem root457Scan9 : scan edge (search edge 3) 4 [3257] 3257 root457Tail9 winner = winner := by rw [root457Tail9, scan, if_pos (by decide)]
theorem root457Scan8 : scan edge (search edge 3) 4 [3257] 3257 root457Tail8 winner = winner :=
  (root457Step8).trans root457Scan9
theorem root457Scan7 : scan edge (search edge 3) 4 [3257] 3257 root457Tail7 winner = winner :=
  (root457Step7).trans root457Scan8
theorem root457Scan6 : scan edge (search edge 3) 4 [3257] 3257 root457Tail6 winner = winner :=
  (root457Step6).trans root457Scan7
theorem root457Scan5 : scan edge (search edge 3) 4 [3257] 3257 root457Tail5 winner = winner :=
  (root457Step5).trans root457Scan6
theorem root457Scan4 : scan edge (search edge 3) 4 [3257] 3257 root457Tail4 winner = winner :=
  (root457Step4).trans root457Scan5
theorem root457Scan3 : scan edge (search edge 3) 4 [3257] 3257 root457Tail3 winner = winner :=
  (root457Step3).trans root457Scan4
theorem root457Scan2 : scan edge (search edge 3) 4 [3257] 3257 root457Tail2 winner = winner :=
  (root457Step2).trans root457Scan3
theorem root457Scan1 : scan edge (search edge 3) 4 [3257] 3257 root457Tail1 winner = winner :=
  (root457Step1).trans root457Scan2
theorem root457Scan0 : scan edge (search edge 3) 4 [3257] 3257 root457Tail0 winner = winner :=
  (root457Step0).trans root457Scan1
set_option maxRecDepth 4096 in
theorem root457Branch : search edge 4 [3257] root457Candidates 3257 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3257] 3257 root457Tail0 winner = winner
  exact root457Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
