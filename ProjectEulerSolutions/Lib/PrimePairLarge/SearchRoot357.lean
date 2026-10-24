import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root357Candidates : List Nat := Data.row2423
set_option maxRecDepth 8192 in
theorem root357Candidates_checked : candidates edge 5 0 2423 tail358 winner.bound = root357Candidates := by
  apply candidates_eq_row edge 5 0 2423 tail358 root357Candidates winner.bound tail358_ordered
  · decide
  · decide
  · funext q
    rfl
def root357Tail45 : List Nat := []
def root357Tail44 : List Nat := 23327 :: root357Tail45
def root357Tail43 : List Nat := 23087 :: root357Tail44
def root357Tail42 : List Nat := 22343 :: root357Tail43
def root357Tail41 : List Nat := 22247 :: root357Tail42
def root357Tail40 : List Nat := 22109 :: root357Tail41
def root357Tail39 : List Nat := 21503 :: root357Tail40
def root357Tail38 : List Nat := 21407 :: root357Tail39
def root357Tail37 : List Nat := 21347 :: root357Tail38
def root357Tail36 : List Nat := 21221 :: root357Tail37
def root357Tail35 : List Nat := 20411 :: root357Tail36
def root357Tail34 : List Nat := 20219 :: root357Tail35
def root357Tail33 : List Nat := 20063 :: root357Tail34
def root357Tail32 : List Nat := 19541 :: root357Tail33
def root357Tail31 : List Nat := 19079 :: root357Tail32
def root357Tail30 : List Nat := 18341 :: root357Tail31
def root357Tail29 : List Nat := 16433 :: root357Tail30
def root357Tail28 : List Nat := 15767 :: root357Tail29
def root357Tail27 : List Nat := 15443 :: root357Tail28
def root357Tail26 : List Nat := 14153 :: root357Tail27
def root357Tail25 : List Nat := 13913 :: root357Tail26
def root357Tail24 : List Nat := 13679 :: root357Tail25
def root357Tail23 : List Nat := 12113 :: root357Tail24
def root357Tail22 : List Nat := 11117 :: root357Tail23
def root357Tail21 : List Nat := 10847 :: root357Tail22
def root357Tail20 : List Nat := 10193 :: root357Tail21
def root357Tail19 : List Nat := 10181 :: root357Tail20
def root357Tail18 : List Nat := 9857 :: root357Tail19
def root357Tail17 : List Nat := 9227 :: root357Tail18
def root357Tail16 : List Nat := 9221 :: root357Tail17
def root357Tail15 : List Nat := 9209 :: root357Tail16
def root357Tail14 : List Nat := 8609 :: root357Tail15
def root357Tail13 : List Nat := 8273 :: root357Tail14
def root357Tail12 : List Nat := 7919 :: root357Tail13
def root357Tail11 : List Nat := 7727 :: root357Tail12
def root357Tail10 : List Nat := 5669 :: root357Tail11
def root357Tail9 : List Nat := 5639 :: root357Tail10
def root357Tail8 : List Nat := 5303 :: root357Tail9
def root357Tail7 : List Nat := 5147 :: root357Tail8
def root357Tail6 : List Nat := 5021 :: root357Tail7
def root357Tail5 : List Nat := 4547 :: root357Tail6
def root357Tail4 : List Nat := 4451 :: root357Tail5
def root357Tail3 : List Nat := 4391 :: root357Tail4
def root357Tail2 : List Nat := 3863 :: root357Tail3
def root357Tail1 : List Nat := 3851 :: root357Tail2
def root357Tail0 : List Nat := 3359 :: root357Tail1
def root357Candidates0 : List Nat := [19079]
set_option maxRecDepth 8192 in
theorem root357Candidates0_checked : candidates edge 4 2423 3359 root357Tail1 winner.bound = root357Candidates0 := by
  unfold candidates
  have row : edge 3359 = (SortedNatTree.contains FastRows.row3359) := by
    change Data.row3359.contains = (SortedNatTree.contains FastRows.row3359)
    exact FastRows.row3359_checked.symm
  rw [row]
  decide
theorem root357Step0 : scan edge (search edge 3) 4 [2423] 2423 root357Tail0 winner =
    scan edge (search edge 3) 4 [2423] 2423 root357Tail1 winner := by
  rw [root357Tail0, scan, if_neg (by decide), root357Candidates0_checked]
  change scan edge (search edge 3) 4 [2423] 2423 root357Tail1
    (if 3 ≤ root357Candidates0.length then search edge 3 [2423, 3359] root357Candidates0 5782 winner else winner) = _
  rw [if_neg (by decide)]
def root357Candidates1 : List Nat := [10181]
set_option maxRecDepth 8192 in
theorem root357Candidates1_checked : candidates edge 4 2423 3851 root357Tail2 winner.bound = root357Candidates1 := by
  unfold candidates
  have row : edge 3851 = (SortedNatTree.contains FastRows.row3851) := by
    change Data.row3851.contains = (SortedNatTree.contains FastRows.row3851)
    exact FastRows.row3851_checked.symm
  rw [row]
  decide
theorem root357Step1 : scan edge (search edge 3) 4 [2423] 2423 root357Tail1 winner =
    scan edge (search edge 3) 4 [2423] 2423 root357Tail2 winner := by
  rw [root357Tail1, scan, if_neg (by decide), root357Candidates1_checked]
  change scan edge (search edge 3) 4 [2423] 2423 root357Tail2
    (if 3 ≤ root357Candidates1.length then search edge 3 [2423, 3851] root357Candidates1 6274 winner else winner) = _
  rw [if_neg (by decide)]
def root357Candidates2 : List Nat := [4451]
set_option maxRecDepth 8192 in
theorem root357Candidates2_checked : candidates edge 4 2423 3863 root357Tail3 winner.bound = root357Candidates2 := by
  unfold candidates
  have row : edge 3863 = (SortedNatTree.contains FastRows.row3863) := by
    change Data.row3863.contains = (SortedNatTree.contains FastRows.row3863)
    exact FastRows.row3863_checked.symm
  rw [row]
  decide
theorem root357Step2 : scan edge (search edge 3) 4 [2423] 2423 root357Tail2 winner =
    scan edge (search edge 3) 4 [2423] 2423 root357Tail3 winner := by
  rw [root357Tail2, scan, if_neg (by decide), root357Candidates2_checked]
  change scan edge (search edge 3) 4 [2423] 2423 root357Tail3
    (if 3 ≤ root357Candidates2.length then search edge 3 [2423, 3863] root357Candidates2 6286 winner else winner) = _
  rw [if_neg (by decide)]
def root357Candidates3 : List Nat := [5639]
set_option maxRecDepth 8192 in
theorem root357Candidates3_checked : candidates edge 4 2423 4391 root357Tail4 winner.bound = root357Candidates3 := by
  unfold candidates
  have row : edge 4391 = (SortedNatTree.contains FastRows.row4391) := by
    change Data.row4391.contains = (SortedNatTree.contains FastRows.row4391)
    exact FastRows.row4391_checked.symm
  rw [row]
  decide
theorem root357Step3 : scan edge (search edge 3) 4 [2423] 2423 root357Tail3 winner =
    scan edge (search edge 3) 4 [2423] 2423 root357Tail4 winner := by
  rw [root357Tail3, scan, if_neg (by decide), root357Candidates3_checked]
  change scan edge (search edge 3) 4 [2423] 2423 root357Tail4
    (if 3 ≤ root357Candidates3.length then search edge 3 [2423, 4391] root357Candidates3 6814 winner else winner) = _
  rw [if_neg (by decide)]
def root357Candidates4 : List Nat := []
set_option maxRecDepth 8192 in
theorem root357Candidates4_checked : candidates edge 4 2423 4451 root357Tail5 winner.bound = root357Candidates4 := by
  unfold candidates
  have row : edge 4451 = (SortedNatTree.contains FastRows.row4451) := by
    change Data.row4451.contains = (SortedNatTree.contains FastRows.row4451)
    exact FastRows.row4451_checked.symm
  rw [row]
  decide
theorem root357Step4 : scan edge (search edge 3) 4 [2423] 2423 root357Tail4 winner =
    scan edge (search edge 3) 4 [2423] 2423 root357Tail5 winner := by
  rw [root357Tail4, scan, if_neg (by decide), root357Candidates4_checked]
  change scan edge (search edge 3) 4 [2423] 2423 root357Tail5
    (if 3 ≤ root357Candidates4.length then search edge 3 [2423, 4451] root357Candidates4 6874 winner else winner) = _
  rw [if_neg (by decide)]
def root357Candidates5 : List Nat := [7919]
set_option maxRecDepth 8192 in
theorem root357Candidates5_checked : candidates edge 4 2423 4547 root357Tail6 winner.bound = root357Candidates5 := by
  unfold candidates
  have row : edge 4547 = (SortedNatTree.contains FastRows.row4547) := by
    change Data.row4547.contains = (SortedNatTree.contains FastRows.row4547)
    exact FastRows.row4547_checked.symm
  rw [row]
  decide
theorem root357Step5 : scan edge (search edge 3) 4 [2423] 2423 root357Tail5 winner =
    scan edge (search edge 3) 4 [2423] 2423 root357Tail6 winner := by
  rw [root357Tail5, scan, if_neg (by decide), root357Candidates5_checked]
  change scan edge (search edge 3) 4 [2423] 2423 root357Tail6
    (if 3 ≤ root357Candidates5.length then search edge 3 [2423, 4547] root357Candidates5 6970 winner else winner) = _
  rw [if_neg (by decide)]
def root357Candidates6 : List Nat := []
set_option maxRecDepth 8192 in
theorem root357Candidates6_checked : candidates edge 4 2423 5021 root357Tail7 winner.bound = root357Candidates6 := by
  unfold candidates
  have row : edge 5021 = (SortedNatTree.contains FastRows.row5021) := by
    change Data.row5021.contains = (SortedNatTree.contains FastRows.row5021)
    exact FastRows.row5021_checked.symm
  rw [row]
  decide
theorem root357Step6 : scan edge (search edge 3) 4 [2423] 2423 root357Tail6 winner =
    scan edge (search edge 3) 4 [2423] 2423 root357Tail7 winner := by
  rw [root357Tail6, scan, if_neg (by decide), root357Candidates6_checked]
  change scan edge (search edge 3) 4 [2423] 2423 root357Tail7
    (if 3 ≤ root357Candidates6.length then search edge 3 [2423, 5021] root357Candidates6 7444 winner else winner) = _
  rw [if_neg (by decide)]
def root357Candidates7 : List Nat := []
set_option maxRecDepth 8192 in
theorem root357Candidates7_checked : candidates edge 4 2423 5147 root357Tail8 winner.bound = root357Candidates7 := by
  unfold candidates
  have row : edge 5147 = (SortedNatTree.contains FastRows.row5147) := by
    change Data.row5147.contains = (SortedNatTree.contains FastRows.row5147)
    exact FastRows.row5147_checked.symm
  rw [row]
  decide
theorem root357Step7 : scan edge (search edge 3) 4 [2423] 2423 root357Tail7 winner =
    scan edge (search edge 3) 4 [2423] 2423 root357Tail8 winner := by
  rw [root357Tail7, scan, if_neg (by decide), root357Candidates7_checked]
  change scan edge (search edge 3) 4 [2423] 2423 root357Tail8
    (if 3 ≤ root357Candidates7.length then search edge 3 [2423, 5147] root357Candidates7 7570 winner else winner) = _
  rw [if_neg (by decide)]
def root357Candidates8 : List Nat := [10181, 15767]
set_option maxRecDepth 8192 in
theorem root357Candidates8_checked : candidates edge 4 2423 5303 root357Tail9 winner.bound = root357Candidates8 := by
  unfold candidates
  have row : edge 5303 = (SortedNatTree.contains FastRows.row5303) := by
    change Data.row5303.contains = (SortedNatTree.contains FastRows.row5303)
    exact FastRows.row5303_checked.symm
  rw [row]
  decide
theorem root357Step8 : scan edge (search edge 3) 4 [2423] 2423 root357Tail8 winner =
    scan edge (search edge 3) 4 [2423] 2423 root357Tail9 winner := by
  rw [root357Tail8, scan, if_neg (by decide), root357Candidates8_checked]
  change scan edge (search edge 3) 4 [2423] 2423 root357Tail9
    (if 3 ≤ root357Candidates8.length then search edge 3 [2423, 5303] root357Candidates8 7726 winner else winner) = _
  rw [if_neg (by decide)]
def root357Candidates9 : List Nat := []
set_option maxRecDepth 8192 in
theorem root357Candidates9_checked : candidates edge 4 2423 5639 root357Tail10 winner.bound = root357Candidates9 := by
  unfold candidates
  have row : edge 5639 = (SortedNatTree.contains FastRows.row5639) := by
    change Data.row5639.contains = (SortedNatTree.contains FastRows.row5639)
    exact FastRows.row5639_checked.symm
  rw [row]
  decide
theorem root357Step9 : scan edge (search edge 3) 4 [2423] 2423 root357Tail9 winner =
    scan edge (search edge 3) 4 [2423] 2423 root357Tail10 winner := by
  rw [root357Tail9, scan, if_neg (by decide), root357Candidates9_checked]
  change scan edge (search edge 3) 4 [2423] 2423 root357Tail10
    (if 3 ≤ root357Candidates9.length then search edge 3 [2423, 5639] root357Candidates9 8062 winner else winner) = _
  rw [if_neg (by decide)]
def root357Candidates10 : List Nat := []
set_option maxRecDepth 8192 in
theorem root357Candidates10_checked : candidates edge 4 2423 5669 root357Tail11 winner.bound = root357Candidates10 := by
  unfold candidates
  have row : edge 5669 = (SortedNatTree.contains FastRows.row5669) := by
    change Data.row5669.contains = (SortedNatTree.contains FastRows.row5669)
    exact FastRows.row5669_checked.symm
  rw [row]
  decide
theorem root357Step10 : scan edge (search edge 3) 4 [2423] 2423 root357Tail10 winner =
    scan edge (search edge 3) 4 [2423] 2423 root357Tail11 winner := by
  rw [root357Tail10, scan, if_neg (by decide), root357Candidates10_checked]
  change scan edge (search edge 3) 4 [2423] 2423 root357Tail11
    (if 3 ≤ root357Candidates10.length then search edge 3 [2423, 5669] root357Candidates10 8092 winner else winner) = _
  rw [if_neg (by decide)]
theorem root357Scan11 : scan edge (search edge 3) 4 [2423] 2423 root357Tail11 winner = winner := by rw [root357Tail11, scan, if_pos (by decide)]
theorem root357Scan10 : scan edge (search edge 3) 4 [2423] 2423 root357Tail10 winner = winner :=
  (root357Step10).trans root357Scan11
theorem root357Scan9 : scan edge (search edge 3) 4 [2423] 2423 root357Tail9 winner = winner :=
  (root357Step9).trans root357Scan10
theorem root357Scan8 : scan edge (search edge 3) 4 [2423] 2423 root357Tail8 winner = winner :=
  (root357Step8).trans root357Scan9
theorem root357Scan7 : scan edge (search edge 3) 4 [2423] 2423 root357Tail7 winner = winner :=
  (root357Step7).trans root357Scan8
theorem root357Scan6 : scan edge (search edge 3) 4 [2423] 2423 root357Tail6 winner = winner :=
  (root357Step6).trans root357Scan7
theorem root357Scan5 : scan edge (search edge 3) 4 [2423] 2423 root357Tail5 winner = winner :=
  (root357Step5).trans root357Scan6
theorem root357Scan4 : scan edge (search edge 3) 4 [2423] 2423 root357Tail4 winner = winner :=
  (root357Step4).trans root357Scan5
theorem root357Scan3 : scan edge (search edge 3) 4 [2423] 2423 root357Tail3 winner = winner :=
  (root357Step3).trans root357Scan4
theorem root357Scan2 : scan edge (search edge 3) 4 [2423] 2423 root357Tail2 winner = winner :=
  (root357Step2).trans root357Scan3
theorem root357Scan1 : scan edge (search edge 3) 4 [2423] 2423 root357Tail1 winner = winner :=
  (root357Step1).trans root357Scan2
theorem root357Scan0 : scan edge (search edge 3) 4 [2423] 2423 root357Tail0 winner = winner :=
  (root357Step0).trans root357Scan1
set_option maxRecDepth 4096 in
theorem root357Branch : search edge 4 [2423] root357Candidates 2423 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2423] 2423 root357Tail0 winner = winner
  exact root357Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
