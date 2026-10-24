import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root503Candidates : List Nat := Data.row3617
set_option maxRecDepth 8192 in
theorem root503Candidates_checked : candidates edge 5 0 3617 tail504 winner.bound = root503Candidates := by
  apply candidates_eq_row edge 5 0 3617 tail504 root503Candidates winner.bound tail504_ordered
  · decide
  · decide
  · funext q
    rfl
def root503Tail42 : List Nat := []
def root503Tail41 : List Nat := 22367 :: root503Tail42
def root503Tail40 : List Nat := 22259 :: root503Tail41
def root503Tail39 : List Nat := 22157 :: root503Tail40
def root503Tail38 : List Nat := 21821 :: root503Tail39
def root503Tail37 : List Nat := 21467 :: root503Tail38
def root503Tail36 : List Nat := 20771 :: root503Tail37
def root503Tail35 : List Nat := 20753 :: root503Tail36
def root503Tail34 : List Nat := 20483 :: root503Tail35
def root503Tail33 : List Nat := 18443 :: root503Tail34
def root503Tail32 : List Nat := 18341 :: root503Tail33
def root503Tail31 : List Nat := 17747 :: root503Tail32
def root503Tail30 : List Nat := 16673 :: root503Tail31
def root503Tail29 : List Nat := 16001 :: root503Tail30
def root503Tail28 : List Nat := 15461 :: root503Tail29
def root503Tail27 : List Nat := 15329 :: root503Tail28
def root503Tail26 : List Nat := 15077 :: root503Tail27
def root503Tail25 : List Nat := 14657 :: root503Tail26
def root503Tail24 : List Nat := 13043 :: root503Tail25
def root503Tail23 : List Nat := 12917 :: root503Tail24
def root503Tail22 : List Nat := 11807 :: root503Tail23
def root503Tail21 : List Nat := 11633 :: root503Tail22
def root503Tail20 : List Nat := 11579 :: root503Tail21
def root503Tail19 : List Nat := 11279 :: root503Tail20
def root503Tail18 : List Nat := 10589 :: root503Tail19
def root503Tail17 : List Nat := 10499 :: root503Tail18
def root503Tail16 : List Nat := 10313 :: root503Tail17
def root503Tail15 : List Nat := 9413 :: root503Tail16
def root503Tail14 : List Nat := 8291 :: root503Tail15
def root503Tail13 : List Nat := 8231 :: root503Tail14
def root503Tail12 : List Nat := 7901 :: root503Tail13
def root503Tail11 : List Nat := 7607 :: root503Tail12
def root503Tail10 : List Nat := 6089 :: root503Tail11
def root503Tail9 : List Nat := 5669 :: root503Tail10
def root503Tail8 : List Nat := 5651 :: root503Tail9
def root503Tail7 : List Nat := 5333 :: root503Tail8
def root503Tail6 : List Nat := 5003 :: root503Tail7
def root503Tail5 : List Nat := 4703 :: root503Tail6
def root503Tail4 : List Nat := 4643 :: root503Tail5
def root503Tail3 : List Nat := 4517 :: root503Tail4
def root503Tail2 : List Nat := 3923 :: root503Tail3
def root503Tail1 : List Nat := 3803 :: root503Tail2
def root503Tail0 : List Nat := 3677 :: root503Tail1
def root503Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root503Candidates0_checked : candidates edge 4 3617 3677 root503Tail1 winner.bound = root503Candidates0 := by
  unfold candidates
  have row : edge 3677 = (SortedNatTree.contains FastRows.row3677) := by
    change Data.row3677.contains = (SortedNatTree.contains FastRows.row3677)
    exact FastRows.row3677_checked.symm
  rw [row]
  decide
theorem root503Step0 : scan edge (search edge 3) 4 [3617] 3617 root503Tail0 winner =
    scan edge (search edge 3) 4 [3617] 3617 root503Tail1 winner := by
  rw [root503Tail0, scan, if_neg (by decide), root503Candidates0_checked]
  change scan edge (search edge 3) 4 [3617] 3617 root503Tail1
    (if 3 ≤ root503Candidates0.length then search edge 3 [3617, 3677] root503Candidates0 7294 winner else winner) = _
  rw [if_neg (by decide)]
def root503Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root503Candidates1_checked : candidates edge 4 3617 3803 root503Tail2 winner.bound = root503Candidates1 := by
  unfold candidates
  have row : edge 3803 = (SortedNatTree.contains FastRows.row3803) := by
    change Data.row3803.contains = (SortedNatTree.contains FastRows.row3803)
    exact FastRows.row3803_checked.symm
  rw [row]
  decide
theorem root503Step1 : scan edge (search edge 3) 4 [3617] 3617 root503Tail1 winner =
    scan edge (search edge 3) 4 [3617] 3617 root503Tail2 winner := by
  rw [root503Tail1, scan, if_neg (by decide), root503Candidates1_checked]
  change scan edge (search edge 3) 4 [3617] 3617 root503Tail2
    (if 3 ≤ root503Candidates1.length then search edge 3 [3617, 3803] root503Candidates1 7420 winner else winner) = _
  rw [if_neg (by decide)]
def root503Candidates2 : List Nat := [4703, 10313]
set_option maxRecDepth 8192 in
theorem root503Candidates2_checked : candidates edge 4 3617 3923 root503Tail3 winner.bound = root503Candidates2 := by
  unfold candidates
  have row : edge 3923 = (SortedNatTree.contains FastRows.row3923) := by
    change Data.row3923.contains = (SortedNatTree.contains FastRows.row3923)
    exact FastRows.row3923_checked.symm
  rw [row]
  decide
theorem root503Step2 : scan edge (search edge 3) 4 [3617] 3617 root503Tail2 winner =
    scan edge (search edge 3) 4 [3617] 3617 root503Tail3 winner := by
  rw [root503Tail2, scan, if_neg (by decide), root503Candidates2_checked]
  change scan edge (search edge 3) 4 [3617] 3617 root503Tail3
    (if 3 ≤ root503Candidates2.length then search edge 3 [3617, 3923] root503Candidates2 7540 winner else winner) = _
  rw [if_neg (by decide)]
def root503Candidates3 : List Nat := [14657]
set_option maxRecDepth 8192 in
theorem root503Candidates3_checked : candidates edge 4 3617 4517 root503Tail4 winner.bound = root503Candidates3 := by
  unfold candidates
  have row : edge 4517 = (SortedNatTree.contains FastRows.row4517) := by
    change Data.row4517.contains = (SortedNatTree.contains FastRows.row4517)
    exact FastRows.row4517_checked.symm
  rw [row]
  decide
theorem root503Step3 : scan edge (search edge 3) 4 [3617] 3617 root503Tail3 winner =
    scan edge (search edge 3) 4 [3617] 3617 root503Tail4 winner := by
  rw [root503Tail3, scan, if_neg (by decide), root503Candidates3_checked]
  change scan edge (search edge 3) 4 [3617] 3617 root503Tail4
    (if 3 ≤ root503Candidates3.length then search edge 3 [3617, 4517] root503Candidates3 8134 winner else winner) = _
  rw [if_neg (by decide)]
def root503Candidates4 : List Nat := [11579]
set_option maxRecDepth 8192 in
theorem root503Candidates4_checked : candidates edge 4 3617 4643 root503Tail5 winner.bound = root503Candidates4 := by
  unfold candidates
  have row : edge 4643 = (SortedNatTree.contains FastRows.row4643) := by
    change Data.row4643.contains = (SortedNatTree.contains FastRows.row4643)
    exact FastRows.row4643_checked.symm
  rw [row]
  decide
theorem root503Step4 : scan edge (search edge 3) 4 [3617] 3617 root503Tail4 winner =
    scan edge (search edge 3) 4 [3617] 3617 root503Tail5 winner := by
  rw [root503Tail4, scan, if_neg (by decide), root503Candidates4_checked]
  change scan edge (search edge 3) 4 [3617] 3617 root503Tail5
    (if 3 ≤ root503Candidates4.length then search edge 3 [3617, 4643] root503Candidates4 8260 winner else winner) = _
  rw [if_neg (by decide)]
def root503Candidates5 : List Nat := []
set_option maxRecDepth 8192 in
theorem root503Candidates5_checked : candidates edge 4 3617 4703 root503Tail6 winner.bound = root503Candidates5 := by
  unfold candidates
  have row : edge 4703 = (SortedNatTree.contains FastRows.row4703) := by
    change Data.row4703.contains = (SortedNatTree.contains FastRows.row4703)
    exact FastRows.row4703_checked.symm
  rw [row]
  decide
theorem root503Step5 : scan edge (search edge 3) 4 [3617] 3617 root503Tail5 winner =
    scan edge (search edge 3) 4 [3617] 3617 root503Tail6 winner := by
  rw [root503Tail5, scan, if_neg (by decide), root503Candidates5_checked]
  change scan edge (search edge 3) 4 [3617] 3617 root503Tail6
    (if 3 ≤ root503Candidates5.length then search edge 3 [3617, 4703] root503Candidates5 8320 winner else winner) = _
  rw [if_neg (by decide)]
def root503Candidates6 : List Nat := [10589]
set_option maxRecDepth 8192 in
theorem root503Candidates6_checked : candidates edge 4 3617 5003 root503Tail7 winner.bound = root503Candidates6 := by
  unfold candidates
  have row : edge 5003 = (SortedNatTree.contains FastRows.row5003) := by
    change Data.row5003.contains = (SortedNatTree.contains FastRows.row5003)
    exact FastRows.row5003_checked.symm
  rw [row]
  decide
theorem root503Step6 : scan edge (search edge 3) 4 [3617] 3617 root503Tail6 winner =
    scan edge (search edge 3) 4 [3617] 3617 root503Tail7 winner := by
  rw [root503Tail6, scan, if_neg (by decide), root503Candidates6_checked]
  change scan edge (search edge 3) 4 [3617] 3617 root503Tail7
    (if 3 ≤ root503Candidates6.length then search edge 3 [3617, 5003] root503Candidates6 8620 winner else winner) = _
  rw [if_neg (by decide)]
def root503Candidates7 : List Nat := [7901]
set_option maxRecDepth 8192 in
theorem root503Candidates7_checked : candidates edge 4 3617 5333 root503Tail8 winner.bound = root503Candidates7 := by
  unfold candidates
  have row : edge 5333 = (SortedNatTree.contains FastRows.row5333) := by
    change Data.row5333.contains = (SortedNatTree.contains FastRows.row5333)
    exact FastRows.row5333_checked.symm
  rw [row]
  decide
theorem root503Step7 : scan edge (search edge 3) 4 [3617] 3617 root503Tail7 winner =
    scan edge (search edge 3) 4 [3617] 3617 root503Tail8 winner := by
  rw [root503Tail7, scan, if_neg (by decide), root503Candidates7_checked]
  change scan edge (search edge 3) 4 [3617] 3617 root503Tail8
    (if 3 ≤ root503Candidates7.length then search edge 3 [3617, 5333] root503Candidates7 8950 winner else winner) = _
  rw [if_neg (by decide)]
theorem root503Scan8 : scan edge (search edge 3) 4 [3617] 3617 root503Tail8 winner = winner := by rw [root503Tail8, scan, if_pos (by decide)]
theorem root503Scan7 : scan edge (search edge 3) 4 [3617] 3617 root503Tail7 winner = winner :=
  (root503Step7).trans root503Scan8
theorem root503Scan6 : scan edge (search edge 3) 4 [3617] 3617 root503Tail6 winner = winner :=
  (root503Step6).trans root503Scan7
theorem root503Scan5 : scan edge (search edge 3) 4 [3617] 3617 root503Tail5 winner = winner :=
  (root503Step5).trans root503Scan6
theorem root503Scan4 : scan edge (search edge 3) 4 [3617] 3617 root503Tail4 winner = winner :=
  (root503Step4).trans root503Scan5
theorem root503Scan3 : scan edge (search edge 3) 4 [3617] 3617 root503Tail3 winner = winner :=
  (root503Step3).trans root503Scan4
theorem root503Scan2 : scan edge (search edge 3) 4 [3617] 3617 root503Tail2 winner = winner :=
  (root503Step2).trans root503Scan3
theorem root503Scan1 : scan edge (search edge 3) 4 [3617] 3617 root503Tail1 winner = winner :=
  (root503Step1).trans root503Scan2
theorem root503Scan0 : scan edge (search edge 3) 4 [3617] 3617 root503Tail0 winner = winner :=
  (root503Step0).trans root503Scan1
set_option maxRecDepth 4096 in
theorem root503Branch : search edge 4 [3617] root503Candidates 3617 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3617] 3617 root503Tail0 winner = winner
  exact root503Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
