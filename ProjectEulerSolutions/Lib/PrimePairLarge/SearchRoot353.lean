import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root353Candidates : List Nat := Data.row2393
set_option maxRecDepth 8192 in
theorem root353Candidates_checked : candidates edge 5 0 2393 tail354 winner.bound = root353Candidates := by
  apply candidates_eq_row edge 5 0 2393 tail354 root353Candidates winner.bound tail354_ordered
  · decide
  · decide
  · funext q
    rfl
def root353Tail46 : List Nat := []
def root353Tail45 : List Nat := 23327 :: root353Tail46
def root353Tail44 : List Nat := 23003 :: root353Tail45
def root353Tail43 : List Nat := 22901 :: root353Tail44
def root353Tail42 : List Nat := 22079 :: root353Tail43
def root353Tail41 : List Nat := 21881 :: root353Tail42
def root353Tail40 : List Nat := 21491 :: root353Tail41
def root353Tail39 : List Nat := 21179 :: root353Tail40
def root353Tail38 : List Nat := 20483 :: root353Tail39
def root353Tail37 : List Nat := 19961 :: root353Tail38
def root353Tail36 : List Nat := 18899 :: root353Tail37
def root353Tail35 : List Nat := 18119 :: root353Tail36
def root353Tail34 : List Nat := 17387 :: root353Tail35
def root353Tail33 : List Nat := 16073 :: root353Tail34
def root353Tail32 : List Nat := 15803 :: root353Tail33
def root353Tail31 : List Nat := 14939 :: root353Tail32
def root353Tail30 : List Nat := 14249 :: root353Tail31
def root353Tail29 : List Nat := 14087 :: root353Tail30
def root353Tail28 : List Nat := 12713 :: root353Tail29
def root353Tail27 : List Nat := 11981 :: root353Tail28
def root353Tail26 : List Nat := 11519 :: root353Tail27
def root353Tail25 : List Nat := 11411 :: root353Tail26
def root353Tail24 : List Nat := 11003 :: root353Tail25
def root353Tail23 : List Nat := 10289 :: root353Tail24
def root353Tail22 : List Nat := 10091 :: root353Tail23
def root353Tail21 : List Nat := 9551 :: root353Tail22
def root353Tail20 : List Nat := 9491 :: root353Tail21
def root353Tail19 : List Nat := 9371 :: root353Tail20
def root353Tail18 : List Nat := 8867 :: root353Tail19
def root353Tail17 : List Nat := 8069 :: root353Tail18
def root353Tail16 : List Nat := 7829 :: root353Tail17
def root353Tail15 : List Nat := 7127 :: root353Tail16
def root353Tail14 : List Nat := 6791 :: root353Tail15
def root353Tail13 : List Nat := 6329 :: root353Tail14
def root353Tail12 : List Nat := 5981 :: root353Tail13
def root353Tail11 : List Nat := 4919 :: root353Tail12
def root353Tail10 : List Nat := 4241 :: root353Tail11
def root353Tail9 : List Nat := 3851 :: root353Tail10
def root353Tail8 : List Nat := 3821 :: root353Tail9
def root353Tail7 : List Nat := 3557 :: root353Tail8
def root353Tail6 : List Nat := 3491 :: root353Tail7
def root353Tail5 : List Nat := 3347 :: root353Tail6
def root353Tail4 : List Nat := 3083 :: root353Tail5
def root353Tail3 : List Nat := 2927 :: root353Tail4
def root353Tail2 : List Nat := 2663 :: root353Tail3
def root353Tail1 : List Nat := 2423 :: root353Tail2
def root353Tail0 : List Nat := 2411 :: root353Tail1
def root353Candidates0 : List Nat := [4241, 4919]
set_option maxRecDepth 8192 in
theorem root353Candidates0_checked : candidates edge 4 2393 2411 root353Tail1 winner.bound = root353Candidates0 := by
  unfold candidates
  have row : edge 2411 = (SortedNatTree.contains FastRows.row2411) := by
    change Data.row2411.contains = (SortedNatTree.contains FastRows.row2411)
    exact FastRows.row2411_checked.symm
  rw [row]
  decide
theorem root353Step0 : scan edge (search edge 3) 4 [2393] 2393 root353Tail0 winner =
    scan edge (search edge 3) 4 [2393] 2393 root353Tail1 winner := by
  rw [root353Tail0, scan, if_neg (by decide), root353Candidates0_checked]
  change scan edge (search edge 3) 4 [2393] 2393 root353Tail1
    (if 3 ≤ root353Candidates0.length then search edge 3 [2393, 2411] root353Candidates0 4804 winner else winner) = _
  rw [if_neg (by decide)]
def root353Candidates1 : List Nat := [3851]
set_option maxRecDepth 8192 in
theorem root353Candidates1_checked : candidates edge 4 2393 2423 root353Tail2 winner.bound = root353Candidates1 := by
  unfold candidates
  have row : edge 2423 = (SortedNatTree.contains FastRows.row2423) := by
    change Data.row2423.contains = (SortedNatTree.contains FastRows.row2423)
    exact FastRows.row2423_checked.symm
  rw [row]
  decide
theorem root353Step1 : scan edge (search edge 3) 4 [2393] 2393 root353Tail1 winner =
    scan edge (search edge 3) 4 [2393] 2393 root353Tail2 winner := by
  rw [root353Tail1, scan, if_neg (by decide), root353Candidates1_checked]
  change scan edge (search edge 3) 4 [2393] 2393 root353Tail2
    (if 3 ≤ root353Candidates1.length then search edge 3 [2393, 2423] root353Candidates1 4816 winner else winner) = _
  rw [if_neg (by decide)]
def root353Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root353Candidates2_checked : candidates edge 4 2393 2663 root353Tail3 winner.bound = root353Candidates2 := by
  unfold candidates
  have row : edge 2663 = (SortedNatTree.contains FastRows.row2663) := by
    change Data.row2663.contains = (SortedNatTree.contains FastRows.row2663)
    exact FastRows.row2663_checked.symm
  rw [row]
  decide
theorem root353Step2 : scan edge (search edge 3) 4 [2393] 2393 root353Tail2 winner =
    scan edge (search edge 3) 4 [2393] 2393 root353Tail3 winner := by
  rw [root353Tail2, scan, if_neg (by decide), root353Candidates2_checked]
  change scan edge (search edge 3) 4 [2393] 2393 root353Tail3
    (if 3 ≤ root353Candidates2.length then search edge 3 [2393, 2663] root353Candidates2 5056 winner else winner) = _
  rw [if_neg (by decide)]
def root353Candidates3 : List Nat := [11519]
set_option maxRecDepth 8192 in
theorem root353Candidates3_checked : candidates edge 4 2393 2927 root353Tail4 winner.bound = root353Candidates3 := by
  unfold candidates
  have row : edge 2927 = (SortedNatTree.contains FastRows.row2927) := by
    change Data.row2927.contains = (SortedNatTree.contains FastRows.row2927)
    exact FastRows.row2927_checked.symm
  rw [row]
  decide
theorem root353Step3 : scan edge (search edge 3) 4 [2393] 2393 root353Tail3 winner =
    scan edge (search edge 3) 4 [2393] 2393 root353Tail4 winner := by
  rw [root353Tail3, scan, if_neg (by decide), root353Candidates3_checked]
  change scan edge (search edge 3) 4 [2393] 2393 root353Tail4
    (if 3 ≤ root353Candidates3.length then search edge 3 [2393, 2927] root353Candidates3 5320 winner else winner) = _
  rw [if_neg (by decide)]
def root353Candidates4 : List Nat := [11519]
set_option maxRecDepth 8192 in
theorem root353Candidates4_checked : candidates edge 4 2393 3083 root353Tail5 winner.bound = root353Candidates4 := by
  unfold candidates
  have row : edge 3083 = (SortedNatTree.contains FastRows.row3083) := by
    change Data.row3083.contains = (SortedNatTree.contains FastRows.row3083)
    exact FastRows.row3083_checked.symm
  rw [row]
  decide
theorem root353Step4 : scan edge (search edge 3) 4 [2393] 2393 root353Tail4 winner =
    scan edge (search edge 3) 4 [2393] 2393 root353Tail5 winner := by
  rw [root353Tail4, scan, if_neg (by decide), root353Candidates4_checked]
  change scan edge (search edge 3) 4 [2393] 2393 root353Tail5
    (if 3 ≤ root353Candidates4.length then search edge 3 [2393, 3083] root353Candidates4 5476 winner else winner) = _
  rw [if_neg (by decide)]
def root353Candidates5 : List Nat := [3821, 11519, 19961]
set_option maxRecDepth 8192 in
theorem root353Candidates5_checked : candidates edge 4 2393 3347 root353Tail6 winner.bound = root353Candidates5 := by
  unfold candidates
  have row : edge 3347 = (SortedNatTree.contains FastRows.row3347) := by
    change Data.row3347.contains = (SortedNatTree.contains FastRows.row3347)
    exact FastRows.row3347_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root353Child5 : search edge 3 [2393, 3347] root353Candidates5 5740 winner = winner := by decide
theorem root353Step5 : scan edge (search edge 3) 4 [2393] 2393 root353Tail5 winner =
    scan edge (search edge 3) 4 [2393] 2393 root353Tail6 winner := by
  rw [root353Tail5, scan, if_neg (by decide), root353Candidates5_checked]
  change scan edge (search edge 3) 4 [2393] 2393 root353Tail6
    (if 3 ≤ root353Candidates5.length then search edge 3 [2393, 3347] root353Candidates5 5740 winner else winner) = _
  rw [if_pos (by decide), root353Child5]
def root353Candidates6 : List Nat := [16073, 18899]
set_option maxRecDepth 8192 in
theorem root353Candidates6_checked : candidates edge 4 2393 3491 root353Tail7 winner.bound = root353Candidates6 := by
  unfold candidates
  have row : edge 3491 = (SortedNatTree.contains FastRows.row3491) := by
    change Data.row3491.contains = (SortedNatTree.contains FastRows.row3491)
    exact FastRows.row3491_checked.symm
  rw [row]
  decide
theorem root353Step6 : scan edge (search edge 3) 4 [2393] 2393 root353Tail6 winner =
    scan edge (search edge 3) 4 [2393] 2393 root353Tail7 winner := by
  rw [root353Tail6, scan, if_neg (by decide), root353Candidates6_checked]
  change scan edge (search edge 3) 4 [2393] 2393 root353Tail7
    (if 3 ≤ root353Candidates6.length then search edge 3 [2393, 3491] root353Candidates6 5884 winner else winner) = _
  rw [if_neg (by decide)]
def root353Candidates7 : List Nat := [6329]
set_option maxRecDepth 8192 in
theorem root353Candidates7_checked : candidates edge 4 2393 3557 root353Tail8 winner.bound = root353Candidates7 := by
  unfold candidates
  have row : edge 3557 = (SortedNatTree.contains FastRows.row3557) := by
    change Data.row3557.contains = (SortedNatTree.contains FastRows.row3557)
    exact FastRows.row3557_checked.symm
  rw [row]
  decide
theorem root353Step7 : scan edge (search edge 3) 4 [2393] 2393 root353Tail7 winner =
    scan edge (search edge 3) 4 [2393] 2393 root353Tail8 winner := by
  rw [root353Tail7, scan, if_neg (by decide), root353Candidates7_checked]
  change scan edge (search edge 3) 4 [2393] 2393 root353Tail8
    (if 3 ≤ root353Candidates7.length then search edge 3 [2393, 3557] root353Candidates7 5950 winner else winner) = _
  rw [if_neg (by decide)]
def root353Candidates8 : List Nat := []
set_option maxRecDepth 8192 in
theorem root353Candidates8_checked : candidates edge 4 2393 3821 root353Tail9 winner.bound = root353Candidates8 := by
  unfold candidates
  have row : edge 3821 = (SortedNatTree.contains FastRows.row3821) := by
    change Data.row3821.contains = (SortedNatTree.contains FastRows.row3821)
    exact FastRows.row3821_checked.symm
  rw [row]
  decide
theorem root353Step8 : scan edge (search edge 3) 4 [2393] 2393 root353Tail8 winner =
    scan edge (search edge 3) 4 [2393] 2393 root353Tail9 winner := by
  rw [root353Tail8, scan, if_neg (by decide), root353Candidates8_checked]
  change scan edge (search edge 3) 4 [2393] 2393 root353Tail9
    (if 3 ≤ root353Candidates8.length then search edge 3 [2393, 3821] root353Candidates8 6214 winner else winner) = _
  rw [if_neg (by decide)]
def root353Candidates9 : List Nat := [9491, 11981]
set_option maxRecDepth 8192 in
theorem root353Candidates9_checked : candidates edge 4 2393 3851 root353Tail10 winner.bound = root353Candidates9 := by
  unfold candidates
  have row : edge 3851 = (SortedNatTree.contains FastRows.row3851) := by
    change Data.row3851.contains = (SortedNatTree.contains FastRows.row3851)
    exact FastRows.row3851_checked.symm
  rw [row]
  decide
theorem root353Step9 : scan edge (search edge 3) 4 [2393] 2393 root353Tail9 winner =
    scan edge (search edge 3) 4 [2393] 2393 root353Tail10 winner := by
  rw [root353Tail9, scan, if_neg (by decide), root353Candidates9_checked]
  change scan edge (search edge 3) 4 [2393] 2393 root353Tail10
    (if 3 ≤ root353Candidates9.length then search edge 3 [2393, 3851] root353Candidates9 6244 winner else winner) = _
  rw [if_neg (by decide)]
def root353Candidates10 : List Nat := [5981, 10289]
set_option maxRecDepth 8192 in
theorem root353Candidates10_checked : candidates edge 4 2393 4241 root353Tail11 winner.bound = root353Candidates10 := by
  unfold candidates
  have row : edge 4241 = (SortedNatTree.contains FastRows.row4241) := by
    change Data.row4241.contains = (SortedNatTree.contains FastRows.row4241)
    exact FastRows.row4241_checked.symm
  rw [row]
  decide
theorem root353Step10 : scan edge (search edge 3) 4 [2393] 2393 root353Tail10 winner =
    scan edge (search edge 3) 4 [2393] 2393 root353Tail11 winner := by
  rw [root353Tail10, scan, if_neg (by decide), root353Candidates10_checked]
  change scan edge (search edge 3) 4 [2393] 2393 root353Tail11
    (if 3 ≤ root353Candidates10.length then search edge 3 [2393, 4241] root353Candidates10 6634 winner else winner) = _
  rw [if_neg (by decide)]
def root353Candidates11 : List Nat := [7829]
set_option maxRecDepth 8192 in
theorem root353Candidates11_checked : candidates edge 4 2393 4919 root353Tail12 winner.bound = root353Candidates11 := by
  unfold candidates
  have row : edge 4919 = (SortedNatTree.contains FastRows.row4919) := by
    change Data.row4919.contains = (SortedNatTree.contains FastRows.row4919)
    exact FastRows.row4919_checked.symm
  rw [row]
  decide
theorem root353Step11 : scan edge (search edge 3) 4 [2393] 2393 root353Tail11 winner =
    scan edge (search edge 3) 4 [2393] 2393 root353Tail12 winner := by
  rw [root353Tail11, scan, if_neg (by decide), root353Candidates11_checked]
  change scan edge (search edge 3) 4 [2393] 2393 root353Tail12
    (if 3 ≤ root353Candidates11.length then search edge 3 [2393, 4919] root353Candidates11 7312 winner else winner) = _
  rw [if_neg (by decide)]
theorem root353Scan12 : scan edge (search edge 3) 4 [2393] 2393 root353Tail12 winner = winner := by rw [root353Tail12, scan, if_pos (by decide)]
theorem root353Scan11 : scan edge (search edge 3) 4 [2393] 2393 root353Tail11 winner = winner :=
  (root353Step11).trans root353Scan12
theorem root353Scan10 : scan edge (search edge 3) 4 [2393] 2393 root353Tail10 winner = winner :=
  (root353Step10).trans root353Scan11
theorem root353Scan9 : scan edge (search edge 3) 4 [2393] 2393 root353Tail9 winner = winner :=
  (root353Step9).trans root353Scan10
theorem root353Scan8 : scan edge (search edge 3) 4 [2393] 2393 root353Tail8 winner = winner :=
  (root353Step8).trans root353Scan9
theorem root353Scan7 : scan edge (search edge 3) 4 [2393] 2393 root353Tail7 winner = winner :=
  (root353Step7).trans root353Scan8
theorem root353Scan6 : scan edge (search edge 3) 4 [2393] 2393 root353Tail6 winner = winner :=
  (root353Step6).trans root353Scan7
theorem root353Scan5 : scan edge (search edge 3) 4 [2393] 2393 root353Tail5 winner = winner :=
  (root353Step5).trans root353Scan6
theorem root353Scan4 : scan edge (search edge 3) 4 [2393] 2393 root353Tail4 winner = winner :=
  (root353Step4).trans root353Scan5
theorem root353Scan3 : scan edge (search edge 3) 4 [2393] 2393 root353Tail3 winner = winner :=
  (root353Step3).trans root353Scan4
theorem root353Scan2 : scan edge (search edge 3) 4 [2393] 2393 root353Tail2 winner = winner :=
  (root353Step2).trans root353Scan3
theorem root353Scan1 : scan edge (search edge 3) 4 [2393] 2393 root353Tail1 winner = winner :=
  (root353Step1).trans root353Scan2
theorem root353Scan0 : scan edge (search edge 3) 4 [2393] 2393 root353Tail0 winner = winner :=
  (root353Step0).trans root353Scan1
set_option maxRecDepth 4096 in
theorem root353Branch : search edge 4 [2393] root353Candidates 2393 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2393] 2393 root353Tail0 winner = winner
  exact root353Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
