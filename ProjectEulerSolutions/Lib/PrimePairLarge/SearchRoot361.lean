import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root361Candidates : List Nat := Data.row2459
set_option maxRecDepth 8192 in
theorem root361Candidates_checked : candidates edge 5 0 2459 tail362 winner.bound = root361Candidates := by
  apply candidates_eq_row edge 5 0 2459 tail362 root361Candidates winner.bound tail362_ordered
  · decide
  · decide
  · funext q
    rfl
def root361Tail42 : List Nat := []
def root361Tail41 : List Nat := 23459 :: root361Tail42
def root361Tail40 : List Nat := 23333 :: root361Tail41
def root361Tail39 : List Nat := 23189 :: root361Tail40
def root361Tail38 : List Nat := 21737 :: root361Tail39
def root361Tail37 : List Nat := 21701 :: root361Tail38
def root361Tail36 : List Nat := 20231 :: root361Tail37
def root361Tail35 : List Nat := 19073 :: root361Tail36
def root361Tail34 : List Nat := 18731 :: root361Tail35
def root361Tail33 : List Nat := 18593 :: root361Tail34
def root361Tail32 : List Nat := 18059 :: root361Tail33
def root361Tail31 : List Nat := 17807 :: root361Tail32
def root361Tail30 : List Nat := 16217 :: root361Tail31
def root361Tail29 : List Nat := 15077 :: root361Tail30
def root361Tail28 : List Nat := 14897 :: root361Tail29
def root361Tail27 : List Nat := 14561 :: root361Tail28
def root361Tail26 : List Nat := 14177 :: root361Tail27
def root361Tail25 : List Nat := 14159 :: root361Tail26
def root361Tail24 : List Nat := 14153 :: root361Tail25
def root361Tail23 : List Nat := 13757 :: root361Tail24
def root361Tail22 : List Nat := 12143 :: root361Tail23
def root361Tail21 : List Nat := 11261 :: root361Tail22
def root361Tail20 : List Nat := 11087 :: root361Tail21
def root361Tail19 : List Nat := 10859 :: root361Tail20
def root361Tail18 : List Nat := 10799 :: root361Tail19
def root361Tail17 : List Nat := 9161 :: root361Tail18
def root361Tail16 : List Nat := 9059 :: root361Tail17
def root361Tail15 : List Nat := 8807 :: root361Tail16
def root361Tail14 : List Nat := 8423 :: root361Tail15
def root361Tail13 : List Nat := 8069 :: root361Tail14
def root361Tail12 : List Nat := 8009 :: root361Tail13
def root361Tail11 : List Nat := 7211 :: root361Tail12
def root361Tail10 : List Nat := 6899 :: root361Tail11
def root361Tail9 : List Nat := 6353 :: root361Tail10
def root361Tail8 : List Nat := 6101 :: root361Tail9
def root361Tail7 : List Nat := 5843 :: root361Tail8
def root361Tail6 : List Nat := 5171 :: root361Tail7
def root361Tail5 : List Nat := 4919 :: root361Tail6
def root361Tail4 : List Nat := 4463 :: root361Tail5
def root361Tail3 : List Nat := 4289 :: root361Tail4
def root361Tail2 : List Nat := 4139 :: root361Tail3
def root361Tail1 : List Nat := 2927 :: root361Tail2
def root361Tail0 : List Nat := 2657 :: root361Tail1
def root361Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root361Candidates0_checked : candidates edge 4 2459 2657 root361Tail1 winner.bound = root361Candidates0 := by
  unfold candidates
  have row : edge 2657 = (SortedNatTree.contains FastRows.row2657) := by
    change Data.row2657.contains = (SortedNatTree.contains FastRows.row2657)
    exact FastRows.row2657_checked.symm
  rw [row]
  decide
theorem root361Step0 : scan edge (search edge 3) 4 [2459] 2459 root361Tail0 winner =
    scan edge (search edge 3) 4 [2459] 2459 root361Tail1 winner := by
  rw [root361Tail0, scan, if_neg (by decide), root361Candidates0_checked]
  change scan edge (search edge 3) 4 [2459] 2459 root361Tail1
    (if 3 ≤ root361Candidates0.length then search edge 3 [2459, 2657] root361Candidates0 5116 winner else winner) = _
  rw [if_neg (by decide)]
def root361Candidates1 : List Nat := [8009]
set_option maxRecDepth 8192 in
theorem root361Candidates1_checked : candidates edge 4 2459 2927 root361Tail2 winner.bound = root361Candidates1 := by
  unfold candidates
  have row : edge 2927 = (SortedNatTree.contains FastRows.row2927) := by
    change Data.row2927.contains = (SortedNatTree.contains FastRows.row2927)
    exact FastRows.row2927_checked.symm
  rw [row]
  decide
theorem root361Step1 : scan edge (search edge 3) 4 [2459] 2459 root361Tail1 winner =
    scan edge (search edge 3) 4 [2459] 2459 root361Tail2 winner := by
  rw [root361Tail1, scan, if_neg (by decide), root361Candidates1_checked]
  change scan edge (search edge 3) 4 [2459] 2459 root361Tail2
    (if 3 ≤ root361Candidates1.length then search edge 3 [2459, 2927] root361Candidates1 5386 winner else winner) = _
  rw [if_neg (by decide)]
def root361Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root361Candidates2_checked : candidates edge 4 2459 4139 root361Tail3 winner.bound = root361Candidates2 := by
  unfold candidates
  have row : edge 4139 = (SortedNatTree.contains FastRows.row4139) := by
    change Data.row4139.contains = (SortedNatTree.contains FastRows.row4139)
    exact FastRows.row4139_checked.symm
  rw [row]
  decide
theorem root361Step2 : scan edge (search edge 3) 4 [2459] 2459 root361Tail2 winner =
    scan edge (search edge 3) 4 [2459] 2459 root361Tail3 winner := by
  rw [root361Tail2, scan, if_neg (by decide), root361Candidates2_checked]
  change scan edge (search edge 3) 4 [2459] 2459 root361Tail3
    (if 3 ≤ root361Candidates2.length then search edge 3 [2459, 4139] root361Candidates2 6598 winner else winner) = _
  rw [if_neg (by decide)]
def root361Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root361Candidates3_checked : candidates edge 4 2459 4289 root361Tail4 winner.bound = root361Candidates3 := by
  unfold candidates
  have row : edge 4289 = (SortedNatTree.contains FastRows.row4289) := by
    change Data.row4289.contains = (SortedNatTree.contains FastRows.row4289)
    exact FastRows.row4289_checked.symm
  rw [row]
  decide
theorem root361Step3 : scan edge (search edge 3) 4 [2459] 2459 root361Tail3 winner =
    scan edge (search edge 3) 4 [2459] 2459 root361Tail4 winner := by
  rw [root361Tail3, scan, if_neg (by decide), root361Candidates3_checked]
  change scan edge (search edge 3) 4 [2459] 2459 root361Tail4
    (if 3 ≤ root361Candidates3.length then search edge 3 [2459, 4289] root361Candidates3 6748 winner else winner) = _
  rw [if_neg (by decide)]
def root361Candidates4 : List Nat := [10859, 12143]
set_option maxRecDepth 8192 in
theorem root361Candidates4_checked : candidates edge 4 2459 4463 root361Tail5 winner.bound = root361Candidates4 := by
  unfold candidates
  have row : edge 4463 = (SortedNatTree.contains FastRows.row4463) := by
    change Data.row4463.contains = (SortedNatTree.contains FastRows.row4463)
    exact FastRows.row4463_checked.symm
  rw [row]
  decide
theorem root361Step4 : scan edge (search edge 3) 4 [2459] 2459 root361Tail4 winner =
    scan edge (search edge 3) 4 [2459] 2459 root361Tail5 winner := by
  rw [root361Tail4, scan, if_neg (by decide), root361Candidates4_checked]
  change scan edge (search edge 3) 4 [2459] 2459 root361Tail5
    (if 3 ≤ root361Candidates4.length then search edge 3 [2459, 4463] root361Candidates4 6922 winner else winner) = _
  rw [if_neg (by decide)]
def root361Candidates5 : List Nat := []
set_option maxRecDepth 8192 in
theorem root361Candidates5_checked : candidates edge 4 2459 4919 root361Tail6 winner.bound = root361Candidates5 := by
  unfold candidates
  have row : edge 4919 = (SortedNatTree.contains FastRows.row4919) := by
    change Data.row4919.contains = (SortedNatTree.contains FastRows.row4919)
    exact FastRows.row4919_checked.symm
  rw [row]
  decide
theorem root361Step5 : scan edge (search edge 3) 4 [2459] 2459 root361Tail5 winner =
    scan edge (search edge 3) 4 [2459] 2459 root361Tail6 winner := by
  rw [root361Tail5, scan, if_neg (by decide), root361Candidates5_checked]
  change scan edge (search edge 3) 4 [2459] 2459 root361Tail6
    (if 3 ≤ root361Candidates5.length then search edge 3 [2459, 4919] root361Candidates5 7378 winner else winner) = _
  rw [if_neg (by decide)]
def root361Candidates6 : List Nat := []
set_option maxRecDepth 8192 in
theorem root361Candidates6_checked : candidates edge 4 2459 5171 root361Tail7 winner.bound = root361Candidates6 := by
  unfold candidates
  have row : edge 5171 = (SortedNatTree.contains FastRows.row5171) := by
    change Data.row5171.contains = (SortedNatTree.contains FastRows.row5171)
    exact FastRows.row5171_checked.symm
  rw [row]
  decide
theorem root361Step6 : scan edge (search edge 3) 4 [2459] 2459 root361Tail6 winner =
    scan edge (search edge 3) 4 [2459] 2459 root361Tail7 winner := by
  rw [root361Tail6, scan, if_neg (by decide), root361Candidates6_checked]
  change scan edge (search edge 3) 4 [2459] 2459 root361Tail7
    (if 3 ≤ root361Candidates6.length then search edge 3 [2459, 5171] root361Candidates6 7630 winner else winner) = _
  rw [if_neg (by decide)]
def root361Candidates7 : List Nat := [16217]
set_option maxRecDepth 8192 in
theorem root361Candidates7_checked : candidates edge 4 2459 5843 root361Tail8 winner.bound = root361Candidates7 := by
  unfold candidates
  have row : edge 5843 = (SortedNatTree.contains FastRows.row5843) := by
    change Data.row5843.contains = (SortedNatTree.contains FastRows.row5843)
    exact FastRows.row5843_checked.symm
  rw [row]
  decide
theorem root361Step7 : scan edge (search edge 3) 4 [2459] 2459 root361Tail7 winner =
    scan edge (search edge 3) 4 [2459] 2459 root361Tail8 winner := by
  rw [root361Tail7, scan, if_neg (by decide), root361Candidates7_checked]
  change scan edge (search edge 3) 4 [2459] 2459 root361Tail8
    (if 3 ≤ root361Candidates7.length then search edge 3 [2459, 5843] root361Candidates7 8302 winner else winner) = _
  rw [if_neg (by decide)]
theorem root361Scan8 : scan edge (search edge 3) 4 [2459] 2459 root361Tail8 winner = winner := by rw [root361Tail8, scan, if_pos (by decide)]
theorem root361Scan7 : scan edge (search edge 3) 4 [2459] 2459 root361Tail7 winner = winner :=
  (root361Step7).trans root361Scan8
theorem root361Scan6 : scan edge (search edge 3) 4 [2459] 2459 root361Tail6 winner = winner :=
  (root361Step6).trans root361Scan7
theorem root361Scan5 : scan edge (search edge 3) 4 [2459] 2459 root361Tail5 winner = winner :=
  (root361Step5).trans root361Scan6
theorem root361Scan4 : scan edge (search edge 3) 4 [2459] 2459 root361Tail4 winner = winner :=
  (root361Step4).trans root361Scan5
theorem root361Scan3 : scan edge (search edge 3) 4 [2459] 2459 root361Tail3 winner = winner :=
  (root361Step3).trans root361Scan4
theorem root361Scan2 : scan edge (search edge 3) 4 [2459] 2459 root361Tail2 winner = winner :=
  (root361Step2).trans root361Scan3
theorem root361Scan1 : scan edge (search edge 3) 4 [2459] 2459 root361Tail1 winner = winner :=
  (root361Step1).trans root361Scan2
theorem root361Scan0 : scan edge (search edge 3) 4 [2459] 2459 root361Tail0 winner = winner :=
  (root361Step0).trans root361Scan1
set_option maxRecDepth 4096 in
theorem root361Branch : search edge 4 [2459] root361Candidates 2459 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2459] 2459 root361Tail0 winner = winner
  exact root361Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
