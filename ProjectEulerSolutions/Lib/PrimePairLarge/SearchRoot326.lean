import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root326Candidates : List Nat := Data.row2207
set_option maxRecDepth 8192 in
theorem root326Candidates_checked : candidates edge 5 0 2207 tail327 winner.bound = root326Candidates := by
  apply candidates_eq_row edge 5 0 2207 tail327 root326Candidates winner.bound tail327_ordered
  · decide
  · decide
  · funext q
    rfl
def root326Tail41 : List Nat := []
def root326Tail40 : List Nat := 23339 :: root326Tail41
def root326Tail39 : List Nat := 22637 :: root326Tail40
def root326Tail38 : List Nat := 22013 :: root326Tail39
def root326Tail37 : List Nat := 21401 :: root326Tail38
def root326Tail36 : List Nat := 20543 :: root326Tail37
def root326Tail35 : List Nat := 20369 :: root326Tail36
def root326Tail34 : List Nat := 19157 :: root326Tail35
def root326Tail33 : List Nat := 19139 :: root326Tail34
def root326Tail32 : List Nat := 18521 :: root326Tail33
def root326Tail31 : List Nat := 16547 :: root326Tail32
def root326Tail30 : List Nat := 16421 :: root326Tail31
def root326Tail29 : List Nat := 16127 :: root326Tail30
def root326Tail28 : List Nat := 16091 :: root326Tail29
def root326Tail27 : List Nat := 15581 :: root326Tail28
def root326Tail26 : List Nat := 15269 :: root326Tail27
def root326Tail25 : List Nat := 15017 :: root326Tail26
def root326Tail24 : List Nat := 14939 :: root326Tail25
def root326Tail23 : List Nat := 14783 :: root326Tail24
def root326Tail22 : List Nat := 12983 :: root326Tail23
def root326Tail21 : List Nat := 12917 :: root326Tail22
def root326Tail20 : List Nat := 12251 :: root326Tail21
def root326Tail19 : List Nat := 12161 :: root326Tail20
def root326Tail18 : List Nat := 11801 :: root326Tail19
def root326Tail17 : List Nat := 10247 :: root326Tail18
def root326Tail16 : List Nat := 9437 :: root326Tail17
def root326Tail15 : List Nat := 9221 :: root326Tail16
def root326Tail14 : List Nat := 9137 :: root326Tail15
def root326Tail13 : List Nat := 8681 :: root326Tail14
def root326Tail12 : List Nat := 8363 :: root326Tail13
def root326Tail11 : List Nat := 7193 :: root326Tail12
def root326Tail10 : List Nat := 6563 :: root326Tail11
def root326Tail9 : List Nat := 6053 :: root326Tail10
def root326Tail8 : List Nat := 5171 :: root326Tail9
def root326Tail7 : List Nat := 4337 :: root326Tail8
def root326Tail6 : List Nat := 4271 :: root326Tail7
def root326Tail5 : List Nat := 3929 :: root326Tail6
def root326Tail4 : List Nat := 3347 :: root326Tail5
def root326Tail3 : List Nat := 2819 :: root326Tail4
def root326Tail2 : List Nat := 2777 :: root326Tail3
def root326Tail1 : List Nat := 2753 :: root326Tail2
def root326Tail0 : List Nat := 2339 :: root326Tail1
def root326Candidates0 : List Nat := [3347, 4337, 14939, 16127]
set_option maxRecDepth 8192 in
theorem root326Candidates0_checked : candidates edge 4 2207 2339 root326Tail1 winner.bound = root326Candidates0 := by
  unfold candidates
  have row : edge 2339 = (SortedNatTree.contains FastRows.row2339) := by
    change Data.row2339.contains = (SortedNatTree.contains FastRows.row2339)
    exact FastRows.row2339_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root326Child0 : search edge 3 [2207, 2339] root326Candidates0 4546 winner = winner := by decide
theorem root326Step0 : scan edge (search edge 3) 4 [2207] 2207 root326Tail0 winner =
    scan edge (search edge 3) 4 [2207] 2207 root326Tail1 winner := by
  rw [root326Tail0, scan, if_neg (by decide), root326Candidates0_checked]
  change scan edge (search edge 3) 4 [2207] 2207 root326Tail1
    (if 3 ≤ root326Candidates0.length then search edge 3 [2207, 2339] root326Candidates0 4546 winner else winner) = _
  rw [if_pos (by decide), root326Child0]
def root326Candidates1 : List Nat := [5171]
set_option maxRecDepth 8192 in
theorem root326Candidates1_checked : candidates edge 4 2207 2753 root326Tail2 winner.bound = root326Candidates1 := by
  unfold candidates
  have row : edge 2753 = (SortedNatTree.contains FastRows.row2753) := by
    change Data.row2753.contains = (SortedNatTree.contains FastRows.row2753)
    exact FastRows.row2753_checked.symm
  rw [row]
  decide
theorem root326Step1 : scan edge (search edge 3) 4 [2207] 2207 root326Tail1 winner =
    scan edge (search edge 3) 4 [2207] 2207 root326Tail2 winner := by
  rw [root326Tail1, scan, if_neg (by decide), root326Candidates1_checked]
  change scan edge (search edge 3) 4 [2207] 2207 root326Tail2
    (if 3 ≤ root326Candidates1.length then search edge 3 [2207, 2753] root326Candidates1 4960 winner else winner) = _
  rw [if_neg (by decide)]
def root326Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root326Candidates2_checked : candidates edge 4 2207 2777 root326Tail3 winner.bound = root326Candidates2 := by
  unfold candidates
  have row : edge 2777 = (SortedNatTree.contains FastRows.row2777) := by
    change Data.row2777.contains = (SortedNatTree.contains FastRows.row2777)
    exact FastRows.row2777_checked.symm
  rw [row]
  decide
theorem root326Step2 : scan edge (search edge 3) 4 [2207] 2207 root326Tail2 winner =
    scan edge (search edge 3) 4 [2207] 2207 root326Tail3 winner := by
  rw [root326Tail2, scan, if_neg (by decide), root326Candidates2_checked]
  change scan edge (search edge 3) 4 [2207] 2207 root326Tail3
    (if 3 ≤ root326Candidates2.length then search edge 3 [2207, 2777] root326Candidates2 4984 winner else winner) = _
  rw [if_neg (by decide)]
def root326Candidates3 : List Nat := [19139]
set_option maxRecDepth 8192 in
theorem root326Candidates3_checked : candidates edge 4 2207 2819 root326Tail4 winner.bound = root326Candidates3 := by
  unfold candidates
  have row : edge 2819 = (SortedNatTree.contains FastRows.row2819) := by
    change Data.row2819.contains = (SortedNatTree.contains FastRows.row2819)
    exact FastRows.row2819_checked.symm
  rw [row]
  decide
theorem root326Step3 : scan edge (search edge 3) 4 [2207] 2207 root326Tail3 winner =
    scan edge (search edge 3) 4 [2207] 2207 root326Tail4 winner := by
  rw [root326Tail3, scan, if_neg (by decide), root326Candidates3_checked]
  change scan edge (search edge 3) 4 [2207] 2207 root326Tail4
    (if 3 ≤ root326Candidates3.length then search edge 3 [2207, 2819] root326Candidates3 5026 winner else winner) = _
  rw [if_neg (by decide)]
def root326Candidates4 : List Nat := []
set_option maxRecDepth 8192 in
theorem root326Candidates4_checked : candidates edge 4 2207 3347 root326Tail5 winner.bound = root326Candidates4 := by
  unfold candidates
  have row : edge 3347 = (SortedNatTree.contains FastRows.row3347) := by
    change Data.row3347.contains = (SortedNatTree.contains FastRows.row3347)
    exact FastRows.row3347_checked.symm
  rw [row]
  decide
theorem root326Step4 : scan edge (search edge 3) 4 [2207] 2207 root326Tail4 winner =
    scan edge (search edge 3) 4 [2207] 2207 root326Tail5 winner := by
  rw [root326Tail4, scan, if_neg (by decide), root326Candidates4_checked]
  change scan edge (search edge 3) 4 [2207] 2207 root326Tail5
    (if 3 ≤ root326Candidates4.length then search edge 3 [2207, 3347] root326Candidates4 5554 winner else winner) = _
  rw [if_neg (by decide)]
def root326Candidates5 : List Nat := [6053, 19139]
set_option maxRecDepth 8192 in
theorem root326Candidates5_checked : candidates edge 4 2207 3929 root326Tail6 winner.bound = root326Candidates5 := by
  unfold candidates
  have row : edge 3929 = (SortedNatTree.contains FastRows.row3929) := by
    change Data.row3929.contains = (SortedNatTree.contains FastRows.row3929)
    exact FastRows.row3929_checked.symm
  rw [row]
  decide
theorem root326Step5 : scan edge (search edge 3) 4 [2207] 2207 root326Tail5 winner =
    scan edge (search edge 3) 4 [2207] 2207 root326Tail6 winner := by
  rw [root326Tail5, scan, if_neg (by decide), root326Candidates5_checked]
  change scan edge (search edge 3) 4 [2207] 2207 root326Tail6
    (if 3 ≤ root326Candidates5.length then search edge 3 [2207, 3929] root326Candidates5 6136 winner else winner) = _
  rw [if_neg (by decide)]
def root326Candidates6 : List Nat := [9137, 10247, 16127]
set_option maxRecDepth 8192 in
theorem root326Candidates6_checked : candidates edge 4 2207 4271 root326Tail7 winner.bound = root326Candidates6 := by
  unfold candidates
  have row : edge 4271 = (SortedNatTree.contains FastRows.row4271) := by
    change Data.row4271.contains = (SortedNatTree.contains FastRows.row4271)
    exact FastRows.row4271_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root326Child6 : search edge 3 [2207, 4271] root326Candidates6 6478 winner = winner := by decide
theorem root326Step6 : scan edge (search edge 3) 4 [2207] 2207 root326Tail6 winner =
    scan edge (search edge 3) 4 [2207] 2207 root326Tail7 winner := by
  rw [root326Tail6, scan, if_neg (by decide), root326Candidates6_checked]
  change scan edge (search edge 3) 4 [2207] 2207 root326Tail7
    (if 3 ≤ root326Candidates6.length then search edge 3 [2207, 4271] root326Candidates6 6478 winner else winner) = _
  rw [if_pos (by decide), root326Child6]
def root326Candidates7 : List Nat := []
set_option maxRecDepth 8192 in
theorem root326Candidates7_checked : candidates edge 4 2207 4337 root326Tail8 winner.bound = root326Candidates7 := by
  unfold candidates
  have row : edge 4337 = (SortedNatTree.contains FastRows.row4337) := by
    change Data.row4337.contains = (SortedNatTree.contains FastRows.row4337)
    exact FastRows.row4337_checked.symm
  rw [row]
  decide
theorem root326Step7 : scan edge (search edge 3) 4 [2207] 2207 root326Tail7 winner =
    scan edge (search edge 3) 4 [2207] 2207 root326Tail8 winner := by
  rw [root326Tail7, scan, if_neg (by decide), root326Candidates7_checked]
  change scan edge (search edge 3) 4 [2207] 2207 root326Tail8
    (if 3 ≤ root326Candidates7.length then search edge 3 [2207, 4337] root326Candidates7 6544 winner else winner) = _
  rw [if_neg (by decide)]
def root326Candidates8 : List Nat := [10247, 15017, 18521]
set_option maxRecDepth 8192 in
theorem root326Candidates8_checked : candidates edge 4 2207 5171 root326Tail9 winner.bound = root326Candidates8 := by
  unfold candidates
  have row : edge 5171 = (SortedNatTree.contains FastRows.row5171) := by
    change Data.row5171.contains = (SortedNatTree.contains FastRows.row5171)
    exact FastRows.row5171_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root326Child8 : search edge 3 [2207, 5171] root326Candidates8 7378 winner = winner := by decide
theorem root326Step8 : scan edge (search edge 3) 4 [2207] 2207 root326Tail8 winner =
    scan edge (search edge 3) 4 [2207] 2207 root326Tail9 winner := by
  rw [root326Tail8, scan, if_neg (by decide), root326Candidates8_checked]
  change scan edge (search edge 3) 4 [2207] 2207 root326Tail9
    (if 3 ≤ root326Candidates8.length then search edge 3 [2207, 5171] root326Candidates8 7378 winner else winner) = _
  rw [if_pos (by decide), root326Child8]
theorem root326Scan9 : scan edge (search edge 3) 4 [2207] 2207 root326Tail9 winner = winner := by rw [root326Tail9, scan, if_pos (by decide)]
theorem root326Scan8 : scan edge (search edge 3) 4 [2207] 2207 root326Tail8 winner = winner :=
  (root326Step8).trans root326Scan9
theorem root326Scan7 : scan edge (search edge 3) 4 [2207] 2207 root326Tail7 winner = winner :=
  (root326Step7).trans root326Scan8
theorem root326Scan6 : scan edge (search edge 3) 4 [2207] 2207 root326Tail6 winner = winner :=
  (root326Step6).trans root326Scan7
theorem root326Scan5 : scan edge (search edge 3) 4 [2207] 2207 root326Tail5 winner = winner :=
  (root326Step5).trans root326Scan6
theorem root326Scan4 : scan edge (search edge 3) 4 [2207] 2207 root326Tail4 winner = winner :=
  (root326Step4).trans root326Scan5
theorem root326Scan3 : scan edge (search edge 3) 4 [2207] 2207 root326Tail3 winner = winner :=
  (root326Step3).trans root326Scan4
theorem root326Scan2 : scan edge (search edge 3) 4 [2207] 2207 root326Tail2 winner = winner :=
  (root326Step2).trans root326Scan3
theorem root326Scan1 : scan edge (search edge 3) 4 [2207] 2207 root326Tail1 winner = winner :=
  (root326Step1).trans root326Scan2
theorem root326Scan0 : scan edge (search edge 3) 4 [2207] 2207 root326Tail0 winner = winner :=
  (root326Step0).trans root326Scan1
set_option maxRecDepth 4096 in
theorem root326Branch : search edge 4 [2207] root326Candidates 2207 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2207] 2207 root326Tail0 winner = winner
  exact root326Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
