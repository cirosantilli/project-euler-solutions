import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root183Candidates : List Nat := Data.row1109
set_option maxRecDepth 8192 in
theorem root183Candidates_checked : candidates edge 5 0 1109 tail184 winner.bound = root183Candidates := by
  apply candidates_eq_row edge 5 0 1109 tail184 root183Candidates winner.bound tail184_ordered
  · decide
  · decide
  · funext q
    rfl
def root183Tail44 : List Nat := []
def root183Tail43 : List Nat := 24509 :: root183Tail44
def root183Tail42 : List Nat := 23537 :: root183Tail43
def root183Tail41 : List Nat := 23279 :: root183Tail42
def root183Tail40 : List Nat := 21611 :: root183Tail41
def root183Tail39 : List Nat := 21347 :: root183Tail40
def root183Tail38 : List Nat := 21179 :: root183Tail39
def root183Tail37 : List Nat := 21101 :: root183Tail38
def root183Tail36 : List Nat := 21017 :: root183Tail37
def root183Tail35 : List Nat := 20261 :: root183Tail36
def root183Tail34 : List Nat := 19463 :: root183Tail35
def root183Tail33 : List Nat := 19157 :: root183Tail34
def root183Tail32 : List Nat := 19121 :: root183Tail33
def root183Tail31 : List Nat := 19079 :: root183Tail32
def root183Tail30 : List Nat := 18839 :: root183Tail31
def root183Tail29 : List Nat := 17981 :: root183Tail30
def root183Tail28 : List Nat := 17483 :: root183Tail29
def root183Tail27 : List Nat := 17351 :: root183Tail28
def root183Tail26 : List Nat := 16103 :: root183Tail27
def root183Tail25 : List Nat := 14957 :: root183Tail26
def root183Tail24 : List Nat := 14411 :: root183Tail25
def root183Tail23 : List Nat := 14087 :: root183Tail24
def root183Tail22 : List Nat := 12959 :: root183Tail23
def root183Tail21 : List Nat := 10691 :: root183Tail22
def root183Tail20 : List Nat := 10589 :: root183Tail21
def root183Tail19 : List Nat := 9923 :: root183Tail20
def root183Tail18 : List Nat := 8663 :: root183Tail19
def root183Tail17 : List Nat := 8369 :: root183Tail18
def root183Tail16 : List Nat := 6719 :: root183Tail17
def root183Tail15 : List Nat := 6521 :: root183Tail16
def root183Tail14 : List Nat := 5501 :: root183Tail15
def root183Tail13 : List Nat := 4871 :: root183Tail14
def root183Tail12 : List Nat := 4493 :: root183Tail13
def root183Tail11 : List Nat := 4289 :: root183Tail12
def root183Tail10 : List Nat := 3923 :: root183Tail11
def root183Tail9 : List Nat := 3917 :: root183Tail10
def root183Tail8 : List Nat := 3491 :: root183Tail9
def root183Tail7 : List Nat := 3209 :: root183Tail8
def root183Tail6 : List Nat := 2999 :: root183Tail7
def root183Tail5 : List Nat := 2441 :: root183Tail6
def root183Tail4 : List Nat := 2069 :: root183Tail5
def root183Tail3 : List Nat := 2027 :: root183Tail4
def root183Tail2 : List Nat := 1889 :: root183Tail3
def root183Tail1 : List Nat := 1229 :: root183Tail2
def root183Tail0 : List Nat := 1181 :: root183Tail1
def root183Candidates0 : List Nat := [3209, 5501, 6719]
set_option maxRecDepth 8192 in
theorem root183Candidates0_checked : candidates edge 4 1109 1181 root183Tail1 winner.bound = root183Candidates0 := by
  unfold candidates
  have row : edge 1181 = (SortedNatTree.contains FastRows.row1181) := by
    change Data.row1181.contains = (SortedNatTree.contains FastRows.row1181)
    exact FastRows.row1181_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root183Child0 : search edge 3 [1109, 1181] root183Candidates0 2290 winner = winner := by decide
theorem root183Step0 : scan edge (search edge 3) 4 [1109] 1109 root183Tail0 winner =
    scan edge (search edge 3) 4 [1109] 1109 root183Tail1 winner := by
  rw [root183Tail0, scan, if_neg (by decide), root183Candidates0_checked]
  change scan edge (search edge 3) 4 [1109] 1109 root183Tail1
    (if 3 ≤ root183Candidates0.length then search edge 3 [1109, 1181] root183Candidates0 2290 winner else winner) = _
  rw [if_pos (by decide), root183Child0]
def root183Candidates1 : List Nat := [10589, 14087, 17351]
set_option maxRecDepth 8192 in
theorem root183Candidates1_checked : candidates edge 4 1109 1229 root183Tail2 winner.bound = root183Candidates1 := by
  unfold candidates
  have row : edge 1229 = (SortedNatTree.contains FastRows.row1229) := by
    change Data.row1229.contains = (SortedNatTree.contains FastRows.row1229)
    exact FastRows.row1229_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root183Child1 : search edge 3 [1109, 1229] root183Candidates1 2338 winner = winner := by decide
theorem root183Step1 : scan edge (search edge 3) 4 [1109] 1109 root183Tail1 winner =
    scan edge (search edge 3) 4 [1109] 1109 root183Tail2 winner := by
  rw [root183Tail1, scan, if_neg (by decide), root183Candidates1_checked]
  change scan edge (search edge 3) 4 [1109] 1109 root183Tail2
    (if 3 ≤ root183Candidates1.length then search edge 3 [1109, 1229] root183Candidates1 2338 winner else winner) = _
  rw [if_pos (by decide), root183Child1]
def root183Candidates2 : List Nat := [2999, 3491]
set_option maxRecDepth 8192 in
theorem root183Candidates2_checked : candidates edge 4 1109 1889 root183Tail3 winner.bound = root183Candidates2 := by
  unfold candidates
  have row : edge 1889 = (SortedNatTree.contains FastRows.row1889) := by
    change Data.row1889.contains = (SortedNatTree.contains FastRows.row1889)
    exact FastRows.row1889_checked.symm
  rw [row]
  decide
theorem root183Step2 : scan edge (search edge 3) 4 [1109] 1109 root183Tail2 winner =
    scan edge (search edge 3) 4 [1109] 1109 root183Tail3 winner := by
  rw [root183Tail2, scan, if_neg (by decide), root183Candidates2_checked]
  change scan edge (search edge 3) 4 [1109] 1109 root183Tail3
    (if 3 ≤ root183Candidates2.length then search edge 3 [1109, 1889] root183Candidates2 2998 winner else winner) = _
  rw [if_neg (by decide)]
def root183Candidates3 : List Nat := [17483]
set_option maxRecDepth 8192 in
theorem root183Candidates3_checked : candidates edge 4 1109 2027 root183Tail4 winner.bound = root183Candidates3 := by
  unfold candidates
  have row : edge 2027 = (SortedNatTree.contains FastRows.row2027) := by
    change Data.row2027.contains = (SortedNatTree.contains FastRows.row2027)
    exact FastRows.row2027_checked.symm
  rw [row]
  decide
theorem root183Step3 : scan edge (search edge 3) 4 [1109] 1109 root183Tail3 winner =
    scan edge (search edge 3) 4 [1109] 1109 root183Tail4 winner := by
  rw [root183Tail3, scan, if_neg (by decide), root183Candidates3_checked]
  change scan edge (search edge 3) 4 [1109] 1109 root183Tail4
    (if 3 ≤ root183Candidates3.length then search edge 3 [1109, 2027] root183Candidates3 3136 winner else winner) = _
  rw [if_neg (by decide)]
def root183Candidates4 : List Nat := [8369, 19463, 21017]
set_option maxRecDepth 8192 in
theorem root183Candidates4_checked : candidates edge 4 1109 2069 root183Tail5 winner.bound = root183Candidates4 := by
  unfold candidates
  have row : edge 2069 = (SortedNatTree.contains FastRows.row2069) := by
    change Data.row2069.contains = (SortedNatTree.contains FastRows.row2069)
    exact FastRows.row2069_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root183Child4 : search edge 3 [1109, 2069] root183Candidates4 3178 winner = winner := by decide
theorem root183Step4 : scan edge (search edge 3) 4 [1109] 1109 root183Tail4 winner =
    scan edge (search edge 3) 4 [1109] 1109 root183Tail5 winner := by
  rw [root183Tail4, scan, if_neg (by decide), root183Candidates4_checked]
  change scan edge (search edge 3) 4 [1109] 1109 root183Tail5
    (if 3 ≤ root183Candidates4.length then search edge 3 [1109, 2069] root183Candidates4 3178 winner else winner) = _
  rw [if_pos (by decide), root183Child4]
def root183Candidates5 : List Nat := [3209, 17351]
set_option maxRecDepth 8192 in
theorem root183Candidates5_checked : candidates edge 4 1109 2441 root183Tail6 winner.bound = root183Candidates5 := by
  unfold candidates
  have row : edge 2441 = (SortedNatTree.contains FastRows.row2441) := by
    change Data.row2441.contains = (SortedNatTree.contains FastRows.row2441)
    exact FastRows.row2441_checked.symm
  rw [row]
  decide
theorem root183Step5 : scan edge (search edge 3) 4 [1109] 1109 root183Tail5 winner =
    scan edge (search edge 3) 4 [1109] 1109 root183Tail6 winner := by
  rw [root183Tail5, scan, if_neg (by decide), root183Candidates5_checked]
  change scan edge (search edge 3) 4 [1109] 1109 root183Tail6
    (if 3 ≤ root183Candidates5.length then search edge 3 [1109, 2441] root183Candidates5 3550 winner else winner) = _
  rw [if_neg (by decide)]
def root183Candidates6 : List Nat := [4871, 14957, 19121]
set_option maxRecDepth 8192 in
theorem root183Candidates6_checked : candidates edge 4 1109 2999 root183Tail7 winner.bound = root183Candidates6 := by
  unfold candidates
  have row : edge 2999 = (SortedNatTree.contains FastRows.row2999) := by
    change Data.row2999.contains = (SortedNatTree.contains FastRows.row2999)
    exact FastRows.row2999_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root183Child6 : search edge 3 [1109, 2999] root183Candidates6 4108 winner = winner := by decide
theorem root183Step6 : scan edge (search edge 3) 4 [1109] 1109 root183Tail6 winner =
    scan edge (search edge 3) 4 [1109] 1109 root183Tail7 winner := by
  rw [root183Tail6, scan, if_neg (by decide), root183Candidates6_checked]
  change scan edge (search edge 3) 4 [1109] 1109 root183Tail7
    (if 3 ≤ root183Candidates6.length then search edge 3 [1109, 2999] root183Candidates6 4108 winner else winner) = _
  rw [if_pos (by decide), root183Child6]
def root183Candidates7 : List Nat := []
set_option maxRecDepth 8192 in
theorem root183Candidates7_checked : candidates edge 4 1109 3209 root183Tail8 winner.bound = root183Candidates7 := by
  unfold candidates
  have row : edge 3209 = (SortedNatTree.contains FastRows.row3209) := by
    change Data.row3209.contains = (SortedNatTree.contains FastRows.row3209)
    exact FastRows.row3209_checked.symm
  rw [row]
  decide
theorem root183Step7 : scan edge (search edge 3) 4 [1109] 1109 root183Tail7 winner =
    scan edge (search edge 3) 4 [1109] 1109 root183Tail8 winner := by
  rw [root183Tail7, scan, if_neg (by decide), root183Candidates7_checked]
  change scan edge (search edge 3) 4 [1109] 1109 root183Tail8
    (if 3 ≤ root183Candidates7.length then search edge 3 [1109, 3209] root183Candidates7 4318 winner else winner) = _
  rw [if_neg (by decide)]
def root183Candidates8 : List Nat := []
set_option maxRecDepth 8192 in
theorem root183Candidates8_checked : candidates edge 4 1109 3491 root183Tail9 winner.bound = root183Candidates8 := by
  unfold candidates
  have row : edge 3491 = (SortedNatTree.contains FastRows.row3491) := by
    change Data.row3491.contains = (SortedNatTree.contains FastRows.row3491)
    exact FastRows.row3491_checked.symm
  rw [row]
  decide
theorem root183Step8 : scan edge (search edge 3) 4 [1109] 1109 root183Tail8 winner =
    scan edge (search edge 3) 4 [1109] 1109 root183Tail9 winner := by
  rw [root183Tail8, scan, if_neg (by decide), root183Candidates8_checked]
  change scan edge (search edge 3) 4 [1109] 1109 root183Tail9
    (if 3 ≤ root183Candidates8.length then search edge 3 [1109, 3491] root183Candidates8 4600 winner else winner) = _
  rw [if_neg (by decide)]
def root183Candidates9 : List Nat := []
set_option maxRecDepth 8192 in
theorem root183Candidates9_checked : candidates edge 4 1109 3917 root183Tail10 winner.bound = root183Candidates9 := by
  unfold candidates
  have row : edge 3917 = (SortedNatTree.contains FastRows.row3917) := by
    change Data.row3917.contains = (SortedNatTree.contains FastRows.row3917)
    exact FastRows.row3917_checked.symm
  rw [row]
  decide
theorem root183Step9 : scan edge (search edge 3) 4 [1109] 1109 root183Tail9 winner =
    scan edge (search edge 3) 4 [1109] 1109 root183Tail10 winner := by
  rw [root183Tail9, scan, if_neg (by decide), root183Candidates9_checked]
  change scan edge (search edge 3) 4 [1109] 1109 root183Tail10
    (if 3 ≤ root183Candidates9.length then search edge 3 [1109, 3917] root183Candidates9 5026 winner else winner) = _
  rw [if_neg (by decide)]
def root183Candidates10 : List Nat := [8663, 14411, 16103]
set_option maxRecDepth 8192 in
theorem root183Candidates10_checked : candidates edge 4 1109 3923 root183Tail11 winner.bound = root183Candidates10 := by
  unfold candidates
  have row : edge 3923 = (SortedNatTree.contains FastRows.row3923) := by
    change Data.row3923.contains = (SortedNatTree.contains FastRows.row3923)
    exact FastRows.row3923_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root183Child10 : search edge 3 [1109, 3923] root183Candidates10 5032 winner = winner := by decide
theorem root183Step10 : scan edge (search edge 3) 4 [1109] 1109 root183Tail10 winner =
    scan edge (search edge 3) 4 [1109] 1109 root183Tail11 winner := by
  rw [root183Tail10, scan, if_neg (by decide), root183Candidates10_checked]
  change scan edge (search edge 3) 4 [1109] 1109 root183Tail11
    (if 3 ≤ root183Candidates10.length then search edge 3 [1109, 3923] root183Candidates10 5032 winner else winner) = _
  rw [if_pos (by decide), root183Child10]
def root183Candidates11 : List Nat := []
set_option maxRecDepth 8192 in
theorem root183Candidates11_checked : candidates edge 4 1109 4289 root183Tail12 winner.bound = root183Candidates11 := by
  unfold candidates
  have row : edge 4289 = (SortedNatTree.contains FastRows.row4289) := by
    change Data.row4289.contains = (SortedNatTree.contains FastRows.row4289)
    exact FastRows.row4289_checked.symm
  rw [row]
  decide
theorem root183Step11 : scan edge (search edge 3) 4 [1109] 1109 root183Tail11 winner =
    scan edge (search edge 3) 4 [1109] 1109 root183Tail12 winner := by
  rw [root183Tail11, scan, if_neg (by decide), root183Candidates11_checked]
  change scan edge (search edge 3) 4 [1109] 1109 root183Tail12
    (if 3 ≤ root183Candidates11.length then search edge 3 [1109, 4289] root183Candidates11 5398 winner else winner) = _
  rw [if_neg (by decide)]
def root183Candidates12 : List Nat := []
set_option maxRecDepth 8192 in
theorem root183Candidates12_checked : candidates edge 4 1109 4493 root183Tail13 winner.bound = root183Candidates12 := by
  unfold candidates
  have row : edge 4493 = (SortedNatTree.contains FastRows.row4493) := by
    change Data.row4493.contains = (SortedNatTree.contains FastRows.row4493)
    exact FastRows.row4493_checked.symm
  rw [row]
  decide
theorem root183Step12 : scan edge (search edge 3) 4 [1109] 1109 root183Tail12 winner =
    scan edge (search edge 3) 4 [1109] 1109 root183Tail13 winner := by
  rw [root183Tail12, scan, if_neg (by decide), root183Candidates12_checked]
  change scan edge (search edge 3) 4 [1109] 1109 root183Tail13
    (if 3 ≤ root183Candidates12.length then search edge 3 [1109, 4493] root183Candidates12 5602 winner else winner) = _
  rw [if_neg (by decide)]
def root183Candidates13 : List Nat := [6719, 19463]
set_option maxRecDepth 8192 in
theorem root183Candidates13_checked : candidates edge 4 1109 4871 root183Tail14 winner.bound = root183Candidates13 := by
  unfold candidates
  have row : edge 4871 = (SortedNatTree.contains FastRows.row4871) := by
    change Data.row4871.contains = (SortedNatTree.contains FastRows.row4871)
    exact FastRows.row4871_checked.symm
  rw [row]
  decide
theorem root183Step13 : scan edge (search edge 3) 4 [1109] 1109 root183Tail13 winner =
    scan edge (search edge 3) 4 [1109] 1109 root183Tail14 winner := by
  rw [root183Tail13, scan, if_neg (by decide), root183Candidates13_checked]
  change scan edge (search edge 3) 4 [1109] 1109 root183Tail14
    (if 3 ≤ root183Candidates13.length then search edge 3 [1109, 4871] root183Candidates13 5980 winner else winner) = _
  rw [if_neg (by decide)]
def root183Candidates14 : List Nat := []
set_option maxRecDepth 8192 in
theorem root183Candidates14_checked : candidates edge 4 1109 5501 root183Tail15 winner.bound = root183Candidates14 := by
  unfold candidates
  have row : edge 5501 = (SortedNatTree.contains FastRows.row5501) := by
    change Data.row5501.contains = (SortedNatTree.contains FastRows.row5501)
    exact FastRows.row5501_checked.symm
  rw [row]
  decide
theorem root183Step14 : scan edge (search edge 3) 4 [1109] 1109 root183Tail14 winner =
    scan edge (search edge 3) 4 [1109] 1109 root183Tail15 winner := by
  rw [root183Tail14, scan, if_neg (by decide), root183Candidates14_checked]
  change scan edge (search edge 3) 4 [1109] 1109 root183Tail15
    (if 3 ≤ root183Candidates14.length then search edge 3 [1109, 5501] root183Candidates14 6610 winner else winner) = _
  rw [if_neg (by decide)]
theorem root183Scan15 : scan edge (search edge 3) 4 [1109] 1109 root183Tail15 winner = winner := by rw [root183Tail15, scan, if_pos (by decide)]
theorem root183Scan14 : scan edge (search edge 3) 4 [1109] 1109 root183Tail14 winner = winner :=
  (root183Step14).trans root183Scan15
theorem root183Scan13 : scan edge (search edge 3) 4 [1109] 1109 root183Tail13 winner = winner :=
  (root183Step13).trans root183Scan14
theorem root183Scan12 : scan edge (search edge 3) 4 [1109] 1109 root183Tail12 winner = winner :=
  (root183Step12).trans root183Scan13
theorem root183Scan11 : scan edge (search edge 3) 4 [1109] 1109 root183Tail11 winner = winner :=
  (root183Step11).trans root183Scan12
theorem root183Scan10 : scan edge (search edge 3) 4 [1109] 1109 root183Tail10 winner = winner :=
  (root183Step10).trans root183Scan11
theorem root183Scan9 : scan edge (search edge 3) 4 [1109] 1109 root183Tail9 winner = winner :=
  (root183Step9).trans root183Scan10
theorem root183Scan8 : scan edge (search edge 3) 4 [1109] 1109 root183Tail8 winner = winner :=
  (root183Step8).trans root183Scan9
theorem root183Scan7 : scan edge (search edge 3) 4 [1109] 1109 root183Tail7 winner = winner :=
  (root183Step7).trans root183Scan8
theorem root183Scan6 : scan edge (search edge 3) 4 [1109] 1109 root183Tail6 winner = winner :=
  (root183Step6).trans root183Scan7
theorem root183Scan5 : scan edge (search edge 3) 4 [1109] 1109 root183Tail5 winner = winner :=
  (root183Step5).trans root183Scan6
theorem root183Scan4 : scan edge (search edge 3) 4 [1109] 1109 root183Tail4 winner = winner :=
  (root183Step4).trans root183Scan5
theorem root183Scan3 : scan edge (search edge 3) 4 [1109] 1109 root183Tail3 winner = winner :=
  (root183Step3).trans root183Scan4
theorem root183Scan2 : scan edge (search edge 3) 4 [1109] 1109 root183Tail2 winner = winner :=
  (root183Step2).trans root183Scan3
theorem root183Scan1 : scan edge (search edge 3) 4 [1109] 1109 root183Tail1 winner = winner :=
  (root183Step1).trans root183Scan2
theorem root183Scan0 : scan edge (search edge 3) 4 [1109] 1109 root183Tail0 winner = winner :=
  (root183Step0).trans root183Scan1
set_option maxRecDepth 4096 in
theorem root183Branch : search edge 4 [1109] root183Candidates 1109 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1109] 1109 root183Tail0 winner = winner
  exact root183Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
