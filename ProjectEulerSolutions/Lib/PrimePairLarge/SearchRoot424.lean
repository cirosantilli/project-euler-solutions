import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root424Candidates : List Nat := Data.row2963
set_option maxRecDepth 8192 in
theorem root424Candidates_checked : candidates edge 5 0 2963 tail425 winner.bound = root424Candidates := by
  apply candidates_eq_row edge 5 0 2963 tail425 root424Candidates winner.bound tail425_ordered
  · decide
  · decide
  · funext q
    rfl
def root424Tail47 : List Nat := []
def root424Tail46 : List Nat := 23057 :: root424Tail47
def root424Tail45 : List Nat := 22619 :: root424Tail46
def root424Tail44 : List Nat := 22541 :: root424Tail45
def root424Tail43 : List Nat := 21737 :: root424Tail44
def root424Tail42 : List Nat := 21401 :: root424Tail43
def root424Tail41 : List Nat := 19427 :: root424Tail42
def root424Tail40 : List Nat := 18251 :: root424Tail41
def root424Tail39 : List Nat := 18233 :: root424Tail40
def root424Tail38 : List Nat := 17627 :: root424Tail39
def root424Tail37 : List Nat := 17393 :: root424Tail38
def root424Tail36 : List Nat := 16871 :: root424Tail37
def root424Tail35 : List Nat := 16673 :: root424Tail36
def root424Tail34 : List Nat := 16193 :: root424Tail35
def root424Tail33 : List Nat := 15629 :: root424Tail34
def root424Tail32 : List Nat := 15161 :: root424Tail33
def root424Tail31 : List Nat := 14939 :: root424Tail32
def root424Tail30 : List Nat := 14783 :: root424Tail31
def root424Tail29 : List Nat := 14303 :: root424Tail30
def root424Tail28 : List Nat := 14033 :: root424Tail29
def root424Tail27 : List Nat := 13931 :: root424Tail28
def root424Tail26 : List Nat := 13883 :: root424Tail27
def root424Tail25 : List Nat := 13679 :: root424Tail26
def root424Tail24 : List Nat := 13187 :: root424Tail25
def root424Tail23 : List Nat := 13103 :: root424Tail24
def root424Tail22 : List Nat := 13049 :: root424Tail23
def root424Tail21 : List Nat := 11867 :: root424Tail22
def root424Tail20 : List Nat := 9851 :: root424Tail21
def root424Tail19 : List Nat := 9371 :: root424Tail20
def root424Tail18 : List Nat := 9011 :: root424Tail19
def root424Tail17 : List Nat := 8663 :: root424Tail18
def root424Tail16 : List Nat := 8111 :: root424Tail17
def root424Tail15 : List Nat := 7691 :: root424Tail16
def root424Tail14 : List Nat := 7607 :: root424Tail15
def root424Tail13 : List Nat := 7187 :: root424Tail14
def root424Tail12 : List Nat := 7151 :: root424Tail13
def root424Tail11 : List Nat := 6689 :: root424Tail12
def root424Tail10 : List Nat := 6449 :: root424Tail11
def root424Tail9 : List Nat := 5861 :: root424Tail10
def root424Tail8 : List Nat := 5693 :: root424Tail9
def root424Tail7 : List Nat := 5381 :: root424Tail8
def root424Tail6 : List Nat := 4919 :: root424Tail7
def root424Tail5 : List Nat := 4691 :: root424Tail6
def root424Tail4 : List Nat := 4421 :: root424Tail5
def root424Tail3 : List Nat := 4391 :: root424Tail4
def root424Tail2 : List Nat := 4079 :: root424Tail3
def root424Tail1 : List Nat := 4019 :: root424Tail2
def root424Tail0 : List Nat := 3557 :: root424Tail1
def root424Candidates0 : List Nat := [13679]
set_option maxRecDepth 8192 in
theorem root424Candidates0_checked : candidates edge 4 2963 3557 root424Tail1 winner.bound = root424Candidates0 := by
  unfold candidates
  have row : edge 3557 = (SortedNatTree.contains FastRows.row3557) := by
    change Data.row3557.contains = (SortedNatTree.contains FastRows.row3557)
    exact FastRows.row3557_checked.symm
  rw [row]
  decide
theorem root424Step0 : scan edge (search edge 3) 4 [2963] 2963 root424Tail0 winner =
    scan edge (search edge 3) 4 [2963] 2963 root424Tail1 winner := by
  rw [root424Tail0, scan, if_neg (by decide), root424Candidates0_checked]
  change scan edge (search edge 3) 4 [2963] 2963 root424Tail1
    (if 3 ≤ root424Candidates0.length then search edge 3 [2963, 3557] root424Candidates0 6520 winner else winner) = _
  rw [if_neg (by decide)]
def root424Candidates1 : List Nat := [15161]
set_option maxRecDepth 8192 in
theorem root424Candidates1_checked : candidates edge 4 2963 4019 root424Tail2 winner.bound = root424Candidates1 := by
  unfold candidates
  have row : edge 4019 = (SortedNatTree.contains FastRows.row4019) := by
    change Data.row4019.contains = (SortedNatTree.contains FastRows.row4019)
    exact FastRows.row4019_checked.symm
  rw [row]
  decide
theorem root424Step1 : scan edge (search edge 3) 4 [2963] 2963 root424Tail1 winner =
    scan edge (search edge 3) 4 [2963] 2963 root424Tail2 winner := by
  rw [root424Tail1, scan, if_neg (by decide), root424Candidates1_checked]
  change scan edge (search edge 3) 4 [2963] 2963 root424Tail2
    (if 3 ≤ root424Candidates1.length then search edge 3 [2963, 4019] root424Candidates1 6982 winner else winner) = _
  rw [if_neg (by decide)]
def root424Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root424Candidates2_checked : candidates edge 4 2963 4079 root424Tail3 winner.bound = root424Candidates2 := by
  unfold candidates
  have row : edge 4079 = (SortedNatTree.contains FastRows.row4079) := by
    change Data.row4079.contains = (SortedNatTree.contains FastRows.row4079)
    exact FastRows.row4079_checked.symm
  rw [row]
  decide
theorem root424Step2 : scan edge (search edge 3) 4 [2963] 2963 root424Tail2 winner =
    scan edge (search edge 3) 4 [2963] 2963 root424Tail3 winner := by
  rw [root424Tail2, scan, if_neg (by decide), root424Candidates2_checked]
  change scan edge (search edge 3) 4 [2963] 2963 root424Tail3
    (if 3 ≤ root424Candidates2.length then search edge 3 [2963, 4079] root424Candidates2 7042 winner else winner) = _
  rw [if_neg (by decide)]
def root424Candidates3 : List Nat := [8111, 16193]
set_option maxRecDepth 8192 in
theorem root424Candidates3_checked : candidates edge 4 2963 4391 root424Tail4 winner.bound = root424Candidates3 := by
  unfold candidates
  have row : edge 4391 = (SortedNatTree.contains FastRows.row4391) := by
    change Data.row4391.contains = (SortedNatTree.contains FastRows.row4391)
    exact FastRows.row4391_checked.symm
  rw [row]
  decide
theorem root424Step3 : scan edge (search edge 3) 4 [2963] 2963 root424Tail3 winner =
    scan edge (search edge 3) 4 [2963] 2963 root424Tail4 winner := by
  rw [root424Tail3, scan, if_neg (by decide), root424Candidates3_checked]
  change scan edge (search edge 3) 4 [2963] 2963 root424Tail4
    (if 3 ≤ root424Candidates3.length then search edge 3 [2963, 4391] root424Candidates3 7354 winner else winner) = _
  rw [if_neg (by decide)]
def root424Candidates4 : List Nat := [15629]
set_option maxRecDepth 8192 in
theorem root424Candidates4_checked : candidates edge 4 2963 4421 root424Tail5 winner.bound = root424Candidates4 := by
  unfold candidates
  have row : edge 4421 = (SortedNatTree.contains FastRows.row4421) := by
    change Data.row4421.contains = (SortedNatTree.contains FastRows.row4421)
    exact FastRows.row4421_checked.symm
  rw [row]
  decide
theorem root424Step4 : scan edge (search edge 3) 4 [2963] 2963 root424Tail4 winner =
    scan edge (search edge 3) 4 [2963] 2963 root424Tail5 winner := by
  rw [root424Tail4, scan, if_neg (by decide), root424Candidates4_checked]
  change scan edge (search edge 3) 4 [2963] 2963 root424Tail5
    (if 3 ≤ root424Candidates4.length then search edge 3 [2963, 4421] root424Candidates4 7384 winner else winner) = _
  rw [if_neg (by decide)]
def root424Candidates5 : List Nat := []
set_option maxRecDepth 8192 in
theorem root424Candidates5_checked : candidates edge 4 2963 4691 root424Tail6 winner.bound = root424Candidates5 := by
  unfold candidates
  have row : edge 4691 = (SortedNatTree.contains FastRows.row4691) := by
    change Data.row4691.contains = (SortedNatTree.contains FastRows.row4691)
    exact FastRows.row4691_checked.symm
  rw [row]
  decide
theorem root424Step5 : scan edge (search edge 3) 4 [2963] 2963 root424Tail5 winner =
    scan edge (search edge 3) 4 [2963] 2963 root424Tail6 winner := by
  rw [root424Tail5, scan, if_neg (by decide), root424Candidates5_checked]
  change scan edge (search edge 3) 4 [2963] 2963 root424Tail6
    (if 3 ≤ root424Candidates5.length then search edge 3 [2963, 4691] root424Candidates5 7654 winner else winner) = _
  rw [if_neg (by decide)]
def root424Candidates6 : List Nat := [7607]
set_option maxRecDepth 8192 in
theorem root424Candidates6_checked : candidates edge 4 2963 4919 root424Tail7 winner.bound = root424Candidates6 := by
  unfold candidates
  have row : edge 4919 = (SortedNatTree.contains FastRows.row4919) := by
    change Data.row4919.contains = (SortedNatTree.contains FastRows.row4919)
    exact FastRows.row4919_checked.symm
  rw [row]
  decide
theorem root424Step6 : scan edge (search edge 3) 4 [2963] 2963 root424Tail6 winner =
    scan edge (search edge 3) 4 [2963] 2963 root424Tail7 winner := by
  rw [root424Tail6, scan, if_neg (by decide), root424Candidates6_checked]
  change scan edge (search edge 3) 4 [2963] 2963 root424Tail7
    (if 3 ≤ root424Candidates6.length then search edge 3 [2963, 4919] root424Candidates6 7882 winner else winner) = _
  rw [if_neg (by decide)]
def root424Candidates7 : List Nat := [9851, 13931]
set_option maxRecDepth 8192 in
theorem root424Candidates7_checked : candidates edge 4 2963 5381 root424Tail8 winner.bound = root424Candidates7 := by
  unfold candidates
  have row : edge 5381 = (SortedNatTree.contains FastRows.row5381) := by
    change Data.row5381.contains = (SortedNatTree.contains FastRows.row5381)
    exact FastRows.row5381_checked.symm
  rw [row]
  decide
theorem root424Step7 : scan edge (search edge 3) 4 [2963] 2963 root424Tail7 winner =
    scan edge (search edge 3) 4 [2963] 2963 root424Tail8 winner := by
  rw [root424Tail7, scan, if_neg (by decide), root424Candidates7_checked]
  change scan edge (search edge 3) 4 [2963] 2963 root424Tail8
    (if 3 ≤ root424Candidates7.length then search edge 3 [2963, 5381] root424Candidates7 8344 winner else winner) = _
  rw [if_neg (by decide)]
def root424Candidates8 : List Nat := [7187, 14033, 15629]
set_option maxRecDepth 8192 in
theorem root424Candidates8_checked : candidates edge 4 2963 5693 root424Tail9 winner.bound = root424Candidates8 := by
  unfold candidates
  have row : edge 5693 = (SortedNatTree.contains FastRows.row5693) := by
    change Data.row5693.contains = (SortedNatTree.contains FastRows.row5693)
    exact FastRows.row5693_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root424Child8 : search edge 3 [2963, 5693] root424Candidates8 8656 winner = winner := by decide
theorem root424Step8 : scan edge (search edge 3) 4 [2963] 2963 root424Tail8 winner =
    scan edge (search edge 3) 4 [2963] 2963 root424Tail9 winner := by
  rw [root424Tail8, scan, if_neg (by decide), root424Candidates8_checked]
  change scan edge (search edge 3) 4 [2963] 2963 root424Tail9
    (if 3 ≤ root424Candidates8.length then search edge 3 [2963, 5693] root424Candidates8 8656 winner else winner) = _
  rw [if_pos (by decide), root424Child8]
theorem root424Scan9 : scan edge (search edge 3) 4 [2963] 2963 root424Tail9 winner = winner := by rw [root424Tail9, scan, if_pos (by decide)]
theorem root424Scan8 : scan edge (search edge 3) 4 [2963] 2963 root424Tail8 winner = winner :=
  (root424Step8).trans root424Scan9
theorem root424Scan7 : scan edge (search edge 3) 4 [2963] 2963 root424Tail7 winner = winner :=
  (root424Step7).trans root424Scan8
theorem root424Scan6 : scan edge (search edge 3) 4 [2963] 2963 root424Tail6 winner = winner :=
  (root424Step6).trans root424Scan7
theorem root424Scan5 : scan edge (search edge 3) 4 [2963] 2963 root424Tail5 winner = winner :=
  (root424Step5).trans root424Scan6
theorem root424Scan4 : scan edge (search edge 3) 4 [2963] 2963 root424Tail4 winner = winner :=
  (root424Step4).trans root424Scan5
theorem root424Scan3 : scan edge (search edge 3) 4 [2963] 2963 root424Tail3 winner = winner :=
  (root424Step3).trans root424Scan4
theorem root424Scan2 : scan edge (search edge 3) 4 [2963] 2963 root424Tail2 winner = winner :=
  (root424Step2).trans root424Scan3
theorem root424Scan1 : scan edge (search edge 3) 4 [2963] 2963 root424Tail1 winner = winner :=
  (root424Step1).trans root424Scan2
theorem root424Scan0 : scan edge (search edge 3) 4 [2963] 2963 root424Tail0 winner = winner :=
  (root424Step0).trans root424Scan1
set_option maxRecDepth 4096 in
theorem root424Branch : search edge 4 [2963] root424Candidates 2963 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2963] 2963 root424Tail0 winner = winner
  exact root424Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
