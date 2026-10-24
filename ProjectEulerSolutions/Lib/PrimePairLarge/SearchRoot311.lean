import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root311Candidates : List Nat := Data.row2083
set_option maxRecDepth 8192 in
theorem root311Candidates_checked : candidates edge 5 0 2083 tail312 winner.bound = root311Candidates := by
  apply candidates_eq_row edge 5 0 2083 tail312 root311Candidates winner.bound tail312_ordered
  · decide
  · decide
  · funext q
    rfl
def root311Tail48 : List Nat := []
def root311Tail47 : List Nat := 23497 :: root311Tail48
def root311Tail46 : List Nat := 23473 :: root311Tail47
def root311Tail45 : List Nat := 22717 :: root311Tail46
def root311Tail44 : List Nat := 22129 :: root311Tail45
def root311Tail43 : List Nat := 21991 :: root311Tail44
def root311Tail42 : List Nat := 20443 :: root311Tail43
def root311Tail41 : List Nat := 20107 :: root311Tail42
def root311Tail40 : List Nat := 19273 :: root311Tail41
def root311Tail39 : List Nat := 18541 :: root311Tail40
def root311Tail38 : List Nat := 18457 :: root311Tail39
def root311Tail37 : List Nat := 18307 :: root311Tail38
def root311Tail36 : List Nat := 18301 :: root311Tail37
def root311Tail35 : List Nat := 18013 :: root311Tail36
def root311Tail34 : List Nat := 17911 :: root311Tail35
def root311Tail33 : List Nat := 17581 :: root311Tail34
def root311Tail32 : List Nat := 17293 :: root311Tail33
def root311Tail31 : List Nat := 15361 :: root311Tail32
def root311Tail30 : List Nat := 14929 :: root311Tail31
def root311Tail29 : List Nat := 14851 :: root311Tail30
def root311Tail28 : List Nat := 14149 :: root311Tail29
def root311Tail27 : List Nat := 13297 :: root311Tail28
def root311Tail26 : List Nat := 12619 :: root311Tail27
def root311Tail25 : List Nat := 12451 :: root311Tail26
def root311Tail24 : List Nat := 12301 :: root311Tail25
def root311Tail23 : List Nat := 11827 :: root311Tail24
def root311Tail22 : List Nat := 11617 :: root311Tail23
def root311Tail21 : List Nat := 11287 :: root311Tail22
def root311Tail20 : List Nat := 11197 :: root311Tail21
def root311Tail19 : List Nat := 10939 :: root311Tail20
def root311Tail18 : List Nat := 9901 :: root311Tail19
def root311Tail17 : List Nat := 9829 :: root311Tail18
def root311Tail16 : List Nat := 9109 :: root311Tail17
def root311Tail15 : List Nat := 8929 :: root311Tail16
def root311Tail14 : List Nat := 8677 :: root311Tail15
def root311Tail13 : List Nat := 8599 :: root311Tail14
def root311Tail12 : List Nat := 8221 :: root311Tail13
def root311Tail11 : List Nat := 7459 :: root311Tail12
def root311Tail10 : List Nat := 6997 :: root311Tail11
def root311Tail9 : List Nat := 6121 :: root311Tail10
def root311Tail8 : List Nat := 5923 :: root311Tail9
def root311Tail7 : List Nat := 5827 :: root311Tail8
def root311Tail6 : List Nat := 5443 :: root311Tail7
def root311Tail5 : List Nat := 4651 :: root311Tail6
def root311Tail4 : List Nat := 4243 :: root311Tail5
def root311Tail3 : List Nat := 4111 :: root311Tail4
def root311Tail2 : List Nat := 3511 :: root311Tail3
def root311Tail1 : List Nat := 3343 :: root311Tail2
def root311Tail0 : List Nat := 2971 :: root311Tail1
def root311Candidates0 : List Nat := [7459]
set_option maxRecDepth 8192 in
theorem root311Candidates0_checked : candidates edge 4 2083 2971 root311Tail1 winner.bound = root311Candidates0 := by
  unfold candidates
  have row : edge 2971 = (SortedNatTree.contains FastRows.row2971) := by
    change Data.row2971.contains = (SortedNatTree.contains FastRows.row2971)
    exact FastRows.row2971_checked.symm
  rw [row]
  decide
theorem root311Step0 : scan edge (search edge 3) 4 [2083] 2083 root311Tail0 winner =
    scan edge (search edge 3) 4 [2083] 2083 root311Tail1 winner := by
  rw [root311Tail0, scan, if_neg (by decide), root311Candidates0_checked]
  change scan edge (search edge 3) 4 [2083] 2083 root311Tail1
    (if 3 ≤ root311Candidates0.length then search edge 3 [2083, 2971] root311Candidates0 5054 winner else winner) = _
  rw [if_neg (by decide)]
def root311Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root311Candidates1_checked : candidates edge 4 2083 3343 root311Tail2 winner.bound = root311Candidates1 := by
  unfold candidates
  have row : edge 3343 = (SortedNatTree.contains FastRows.row3343) := by
    change Data.row3343.contains = (SortedNatTree.contains FastRows.row3343)
    exact FastRows.row3343_checked.symm
  rw [row]
  decide
theorem root311Step1 : scan edge (search edge 3) 4 [2083] 2083 root311Tail1 winner =
    scan edge (search edge 3) 4 [2083] 2083 root311Tail2 winner := by
  rw [root311Tail1, scan, if_neg (by decide), root311Candidates1_checked]
  change scan edge (search edge 3) 4 [2083] 2083 root311Tail2
    (if 3 ≤ root311Candidates1.length then search edge 3 [2083, 3343] root311Candidates1 5426 winner else winner) = _
  rw [if_neg (by decide)]
def root311Candidates2 : List Nat := [12451]
set_option maxRecDepth 8192 in
theorem root311Candidates2_checked : candidates edge 4 2083 3511 root311Tail3 winner.bound = root311Candidates2 := by
  unfold candidates
  have row : edge 3511 = (SortedNatTree.contains FastRows.row3511) := by
    change Data.row3511.contains = (SortedNatTree.contains FastRows.row3511)
    exact FastRows.row3511_checked.symm
  rw [row]
  decide
theorem root311Step2 : scan edge (search edge 3) 4 [2083] 2083 root311Tail2 winner =
    scan edge (search edge 3) 4 [2083] 2083 root311Tail3 winner := by
  rw [root311Tail2, scan, if_neg (by decide), root311Candidates2_checked]
  change scan edge (search edge 3) 4 [2083] 2083 root311Tail3
    (if 3 ≤ root311Candidates2.length then search edge 3 [2083, 3511] root311Candidates2 5594 winner else winner) = _
  rw [if_neg (by decide)]
def root311Candidates3 : List Nat := [18457]
set_option maxRecDepth 8192 in
theorem root311Candidates3_checked : candidates edge 4 2083 4111 root311Tail4 winner.bound = root311Candidates3 := by
  unfold candidates
  have row : edge 4111 = (SortedNatTree.contains FastRows.row4111) := by
    change Data.row4111.contains = (SortedNatTree.contains FastRows.row4111)
    exact FastRows.row4111_checked.symm
  rw [row]
  decide
theorem root311Step3 : scan edge (search edge 3) 4 [2083] 2083 root311Tail3 winner =
    scan edge (search edge 3) 4 [2083] 2083 root311Tail4 winner := by
  rw [root311Tail3, scan, if_neg (by decide), root311Candidates3_checked]
  change scan edge (search edge 3) 4 [2083] 2083 root311Tail4
    (if 3 ≤ root311Candidates3.length then search edge 3 [2083, 4111] root311Candidates3 6194 winner else winner) = _
  rw [if_neg (by decide)]
def root311Candidates4 : List Nat := [6997, 11197, 15361]
set_option maxRecDepth 8192 in
theorem root311Candidates4_checked : candidates edge 4 2083 4243 root311Tail5 winner.bound = root311Candidates4 := by
  unfold candidates
  have row : edge 4243 = (SortedNatTree.contains FastRows.row4243) := by
    change Data.row4243.contains = (SortedNatTree.contains FastRows.row4243)
    exact FastRows.row4243_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root311Child4 : search edge 3 [2083, 4243] root311Candidates4 6326 winner = winner := by decide
theorem root311Step4 : scan edge (search edge 3) 4 [2083] 2083 root311Tail4 winner =
    scan edge (search edge 3) 4 [2083] 2083 root311Tail5 winner := by
  rw [root311Tail4, scan, if_neg (by decide), root311Candidates4_checked]
  change scan edge (search edge 3) 4 [2083] 2083 root311Tail5
    (if 3 ≤ root311Candidates4.length then search edge 3 [2083, 4243] root311Candidates4 6326 winner else winner) = _
  rw [if_pos (by decide), root311Child4]
def root311Candidates5 : List Nat := [18541]
set_option maxRecDepth 8192 in
theorem root311Candidates5_checked : candidates edge 4 2083 4651 root311Tail6 winner.bound = root311Candidates5 := by
  unfold candidates
  have row : edge 4651 = (SortedNatTree.contains FastRows.row4651) := by
    change Data.row4651.contains = (SortedNatTree.contains FastRows.row4651)
    exact FastRows.row4651_checked.symm
  rw [row]
  decide
theorem root311Step5 : scan edge (search edge 3) 4 [2083] 2083 root311Tail5 winner =
    scan edge (search edge 3) 4 [2083] 2083 root311Tail6 winner := by
  rw [root311Tail5, scan, if_neg (by decide), root311Candidates5_checked]
  change scan edge (search edge 3) 4 [2083] 2083 root311Tail6
    (if 3 ≤ root311Candidates5.length then search edge 3 [2083, 4651] root311Candidates5 6734 winner else winner) = _
  rw [if_neg (by decide)]
def root311Candidates6 : List Nat := [5923]
set_option maxRecDepth 8192 in
theorem root311Candidates6_checked : candidates edge 4 2083 5443 root311Tail7 winner.bound = root311Candidates6 := by
  unfold candidates
  have row : edge 5443 = (SortedNatTree.contains FastRows.row5443) := by
    change Data.row5443.contains = (SortedNatTree.contains FastRows.row5443)
    exact FastRows.row5443_checked.symm
  rw [row]
  decide
theorem root311Step6 : scan edge (search edge 3) 4 [2083] 2083 root311Tail6 winner =
    scan edge (search edge 3) 4 [2083] 2083 root311Tail7 winner := by
  rw [root311Tail6, scan, if_neg (by decide), root311Candidates6_checked]
  change scan edge (search edge 3) 4 [2083] 2083 root311Tail7
    (if 3 ≤ root311Candidates6.length then search edge 3 [2083, 5443] root311Candidates6 7526 winner else winner) = _
  rw [if_neg (by decide)]
def root311Candidates7 : List Nat := []
set_option maxRecDepth 8192 in
theorem root311Candidates7_checked : candidates edge 4 2083 5827 root311Tail8 winner.bound = root311Candidates7 := by
  unfold candidates
  have row : edge 5827 = (SortedNatTree.contains FastRows.row5827) := by
    change Data.row5827.contains = (SortedNatTree.contains FastRows.row5827)
    exact FastRows.row5827_checked.symm
  rw [row]
  decide
theorem root311Step7 : scan edge (search edge 3) 4 [2083] 2083 root311Tail7 winner =
    scan edge (search edge 3) 4 [2083] 2083 root311Tail8 winner := by
  rw [root311Tail7, scan, if_neg (by decide), root311Candidates7_checked]
  change scan edge (search edge 3) 4 [2083] 2083 root311Tail8
    (if 3 ≤ root311Candidates7.length then search edge 3 [2083, 5827] root311Candidates7 7910 winner else winner) = _
  rw [if_neg (by decide)]
def root311Candidates8 : List Nat := [8929, 14851]
set_option maxRecDepth 8192 in
theorem root311Candidates8_checked : candidates edge 4 2083 5923 root311Tail9 winner.bound = root311Candidates8 := by
  unfold candidates
  have row : edge 5923 = (SortedNatTree.contains FastRows.row5923) := by
    change Data.row5923.contains = (SortedNatTree.contains FastRows.row5923)
    exact FastRows.row5923_checked.symm
  rw [row]
  decide
theorem root311Step8 : scan edge (search edge 3) 4 [2083] 2083 root311Tail8 winner =
    scan edge (search edge 3) 4 [2083] 2083 root311Tail9 winner := by
  rw [root311Tail8, scan, if_neg (by decide), root311Candidates8_checked]
  change scan edge (search edge 3) 4 [2083] 2083 root311Tail9
    (if 3 ≤ root311Candidates8.length then search edge 3 [2083, 5923] root311Candidates8 8006 winner else winner) = _
  rw [if_neg (by decide)]
theorem root311Scan9 : scan edge (search edge 3) 4 [2083] 2083 root311Tail9 winner = winner := by rw [root311Tail9, scan, if_pos (by decide)]
theorem root311Scan8 : scan edge (search edge 3) 4 [2083] 2083 root311Tail8 winner = winner :=
  (root311Step8).trans root311Scan9
theorem root311Scan7 : scan edge (search edge 3) 4 [2083] 2083 root311Tail7 winner = winner :=
  (root311Step7).trans root311Scan8
theorem root311Scan6 : scan edge (search edge 3) 4 [2083] 2083 root311Tail6 winner = winner :=
  (root311Step6).trans root311Scan7
theorem root311Scan5 : scan edge (search edge 3) 4 [2083] 2083 root311Tail5 winner = winner :=
  (root311Step5).trans root311Scan6
theorem root311Scan4 : scan edge (search edge 3) 4 [2083] 2083 root311Tail4 winner = winner :=
  (root311Step4).trans root311Scan5
theorem root311Scan3 : scan edge (search edge 3) 4 [2083] 2083 root311Tail3 winner = winner :=
  (root311Step3).trans root311Scan4
theorem root311Scan2 : scan edge (search edge 3) 4 [2083] 2083 root311Tail2 winner = winner :=
  (root311Step2).trans root311Scan3
theorem root311Scan1 : scan edge (search edge 3) 4 [2083] 2083 root311Tail1 winner = winner :=
  (root311Step1).trans root311Scan2
theorem root311Scan0 : scan edge (search edge 3) 4 [2083] 2083 root311Tail0 winner = winner :=
  (root311Step0).trans root311Scan1
set_option maxRecDepth 4096 in
theorem root311Branch : search edge 4 [2083] root311Candidates 2083 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2083] 2083 root311Tail0 winner = winner
  exact root311Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
