import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root296Candidates : List Nat := Data.row1979
set_option maxRecDepth 8192 in
theorem root296Candidates_checked : candidates edge 5 0 1979 tail297 winner.bound = root296Candidates := by
  apply candidates_eq_row edge 5 0 1979 tail297 root296Candidates winner.bound tail297_ordered
  · decide
  · decide
  · funext q
    rfl
def root296Tail47 : List Nat := []
def root296Tail46 : List Nat := 24029 :: root296Tail47
def root296Tail45 : List Nat := 23549 :: root296Tail46
def root296Tail44 : List Nat := 23021 :: root296Tail45
def root296Tail43 : List Nat := 22469 :: root296Tail44
def root296Tail42 : List Nat := 22367 :: root296Tail43
def root296Tail41 : List Nat := 22037 :: root296Tail42
def root296Tail40 : List Nat := 21377 :: root296Tail41
def root296Tail39 : List Nat := 20249 :: root296Tail40
def root296Tail38 : List Nat := 19949 :: root296Tail39
def root296Tail37 : List Nat := 19157 :: root296Tail38
def root296Tail36 : List Nat := 18731 :: root296Tail37
def root296Tail35 : List Nat := 18047 :: root296Tail36
def root296Tail34 : List Nat := 17957 :: root296Tail35
def root296Tail33 : List Nat := 17669 :: root296Tail34
def root296Tail32 : List Nat := 16661 :: root296Tail33
def root296Tail31 : List Nat := 15083 :: root296Tail32
def root296Tail30 : List Nat := 14957 :: root296Tail31
def root296Tail29 : List Nat := 14537 :: root296Tail30
def root296Tail28 : List Nat := 14489 :: root296Tail29
def root296Tail27 : List Nat := 14303 :: root296Tail28
def root296Tail26 : List Nat := 13721 :: root296Tail27
def root296Tail25 : List Nat := 13523 :: root296Tail26
def root296Tail24 : List Nat := 13217 :: root296Tail25
def root296Tail23 : List Nat := 13187 :: root296Tail24
def root296Tail22 : List Nat := 13049 :: root296Tail23
def root296Tail21 : List Nat := 12941 :: root296Tail22
def root296Tail20 : List Nat := 12647 :: root296Tail21
def root296Tail19 : List Nat := 12437 :: root296Tail20
def root296Tail18 : List Nat := 11279 :: root296Tail19
def root296Tail17 : List Nat := 11117 :: root296Tail18
def root296Tail16 : List Nat := 9833 :: root296Tail17
def root296Tail15 : List Nat := 9491 :: root296Tail16
def root296Tail14 : List Nat := 9413 :: root296Tail15
def root296Tail13 : List Nat := 9323 :: root296Tail14
def root296Tail12 : List Nat := 9137 :: root296Tail13
def root296Tail11 : List Nat := 8297 :: root296Tail12
def root296Tail10 : List Nat := 8291 :: root296Tail11
def root296Tail9 : List Nat := 7433 :: root296Tail10
def root296Tail8 : List Nat := 7187 :: root296Tail9
def root296Tail7 : List Nat := 6869 :: root296Tail8
def root296Tail6 : List Nat := 4937 :: root296Tail7
def root296Tail5 : List Nat := 4241 :: root296Tail6
def root296Tail4 : List Nat := 3911 :: root296Tail5
def root296Tail3 : List Nat := 2999 :: root296Tail4
def root296Tail2 : List Nat := 2693 :: root296Tail3
def root296Tail1 : List Nat := 2207 :: root296Tail2
def root296Tail0 : List Nat := 2039 :: root296Tail1
def root296Candidates0 : List Nat := [4937, 12941]
set_option maxRecDepth 8192 in
theorem root296Candidates0_checked : candidates edge 4 1979 2039 root296Tail1 winner.bound = root296Candidates0 := by
  unfold candidates
  have row : edge 2039 = (SortedNatTree.contains FastRows.row2039) := by
    change Data.row2039.contains = (SortedNatTree.contains FastRows.row2039)
    exact FastRows.row2039_checked.symm
  rw [row]
  decide
theorem root296Step0 : scan edge (search edge 3) 4 [1979] 1979 root296Tail0 winner =
    scan edge (search edge 3) 4 [1979] 1979 root296Tail1 winner := by
  rw [root296Tail0, scan, if_neg (by decide), root296Candidates0_checked]
  change scan edge (search edge 3) 4 [1979] 1979 root296Tail1
    (if 3 ≤ root296Candidates0.length then search edge 3 [1979, 2039] root296Candidates0 4018 winner else winner) = _
  rw [if_neg (by decide)]
def root296Candidates1 : List Nat := [9137, 19157]
set_option maxRecDepth 8192 in
theorem root296Candidates1_checked : candidates edge 4 1979 2207 root296Tail2 winner.bound = root296Candidates1 := by
  unfold candidates
  have row : edge 2207 = (SortedNatTree.contains FastRows.row2207) := by
    change Data.row2207.contains = (SortedNatTree.contains FastRows.row2207)
    exact FastRows.row2207_checked.symm
  rw [row]
  decide
theorem root296Step1 : scan edge (search edge 3) 4 [1979] 1979 root296Tail1 winner =
    scan edge (search edge 3) 4 [1979] 1979 root296Tail2 winner := by
  rw [root296Tail1, scan, if_neg (by decide), root296Candidates1_checked]
  change scan edge (search edge 3) 4 [1979] 1979 root296Tail2
    (if 3 ≤ root296Candidates1.length then search edge 3 [1979, 2207] root296Candidates1 4186 winner else winner) = _
  rw [if_neg (by decide)]
def root296Candidates2 : List Nat := [9833, 13217]
set_option maxRecDepth 8192 in
theorem root296Candidates2_checked : candidates edge 4 1979 2693 root296Tail3 winner.bound = root296Candidates2 := by
  unfold candidates
  have row : edge 2693 = (SortedNatTree.contains FastRows.row2693) := by
    change Data.row2693.contains = (SortedNatTree.contains FastRows.row2693)
    exact FastRows.row2693_checked.symm
  rw [row]
  decide
theorem root296Step2 : scan edge (search edge 3) 4 [1979] 1979 root296Tail2 winner =
    scan edge (search edge 3) 4 [1979] 1979 root296Tail3 winner := by
  rw [root296Tail2, scan, if_neg (by decide), root296Candidates2_checked]
  change scan edge (search edge 3) 4 [1979] 1979 root296Tail3
    (if 3 ≤ root296Candidates2.length then search edge 3 [1979, 2693] root296Candidates2 4672 winner else winner) = _
  rw [if_neg (by decide)]
def root296Candidates3 : List Nat := [14957]
set_option maxRecDepth 8192 in
theorem root296Candidates3_checked : candidates edge 4 1979 2999 root296Tail4 winner.bound = root296Candidates3 := by
  unfold candidates
  have row : edge 2999 = (SortedNatTree.contains FastRows.row2999) := by
    change Data.row2999.contains = (SortedNatTree.contains FastRows.row2999)
    exact FastRows.row2999_checked.symm
  rw [row]
  decide
theorem root296Step3 : scan edge (search edge 3) 4 [1979] 1979 root296Tail3 winner =
    scan edge (search edge 3) 4 [1979] 1979 root296Tail4 winner := by
  rw [root296Tail3, scan, if_neg (by decide), root296Candidates3_checked]
  change scan edge (search edge 3) 4 [1979] 1979 root296Tail4
    (if 3 ≤ root296Candidates3.length then search edge 3 [1979, 2999] root296Candidates3 4978 winner else winner) = _
  rw [if_neg (by decide)]
def root296Candidates4 : List Nat := [13523, 14303, 14957]
set_option maxRecDepth 8192 in
theorem root296Candidates4_checked : candidates edge 4 1979 3911 root296Tail5 winner.bound = root296Candidates4 := by
  unfold candidates
  have row : edge 3911 = (SortedNatTree.contains FastRows.row3911) := by
    change Data.row3911.contains = (SortedNatTree.contains FastRows.row3911)
    exact FastRows.row3911_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root296Child4 : search edge 3 [1979, 3911] root296Candidates4 5890 winner = winner := by decide
theorem root296Step4 : scan edge (search edge 3) 4 [1979] 1979 root296Tail4 winner =
    scan edge (search edge 3) 4 [1979] 1979 root296Tail5 winner := by
  rw [root296Tail4, scan, if_neg (by decide), root296Candidates4_checked]
  change scan edge (search edge 3) 4 [1979] 1979 root296Tail5
    (if 3 ≤ root296Candidates4.length then search edge 3 [1979, 3911] root296Candidates4 5890 winner else winner) = _
  rw [if_pos (by decide), root296Child4]
def root296Candidates5 : List Nat := [9323, 9413]
set_option maxRecDepth 8192 in
theorem root296Candidates5_checked : candidates edge 4 1979 4241 root296Tail6 winner.bound = root296Candidates5 := by
  unfold candidates
  have row : edge 4241 = (SortedNatTree.contains FastRows.row4241) := by
    change Data.row4241.contains = (SortedNatTree.contains FastRows.row4241)
    exact FastRows.row4241_checked.symm
  rw [row]
  decide
theorem root296Step5 : scan edge (search edge 3) 4 [1979] 1979 root296Tail5 winner =
    scan edge (search edge 3) 4 [1979] 1979 root296Tail6 winner := by
  rw [root296Tail5, scan, if_neg (by decide), root296Candidates5_checked]
  change scan edge (search edge 3) 4 [1979] 1979 root296Tail6
    (if 3 ≤ root296Candidates5.length then search edge 3 [1979, 4241] root296Candidates5 6220 winner else winner) = _
  rw [if_neg (by decide)]
def root296Candidates6 : List Nat := [13049, 13523]
set_option maxRecDepth 8192 in
theorem root296Candidates6_checked : candidates edge 4 1979 4937 root296Tail7 winner.bound = root296Candidates6 := by
  unfold candidates
  have row : edge 4937 = (SortedNatTree.contains FastRows.row4937) := by
    change Data.row4937.contains = (SortedNatTree.contains FastRows.row4937)
    exact FastRows.row4937_checked.symm
  rw [row]
  decide
theorem root296Step6 : scan edge (search edge 3) 4 [1979] 1979 root296Tail6 winner =
    scan edge (search edge 3) 4 [1979] 1979 root296Tail7 winner := by
  rw [root296Tail6, scan, if_neg (by decide), root296Candidates6_checked]
  change scan edge (search edge 3) 4 [1979] 1979 root296Tail7
    (if 3 ≤ root296Candidates6.length then search edge 3 [1979, 4937] root296Candidates6 6916 winner else winner) = _
  rw [if_neg (by decide)]
theorem root296Scan7 : scan edge (search edge 3) 4 [1979] 1979 root296Tail7 winner = winner := by rw [root296Tail7, scan, if_pos (by decide)]
theorem root296Scan6 : scan edge (search edge 3) 4 [1979] 1979 root296Tail6 winner = winner :=
  (root296Step6).trans root296Scan7
theorem root296Scan5 : scan edge (search edge 3) 4 [1979] 1979 root296Tail5 winner = winner :=
  (root296Step5).trans root296Scan6
theorem root296Scan4 : scan edge (search edge 3) 4 [1979] 1979 root296Tail4 winner = winner :=
  (root296Step4).trans root296Scan5
theorem root296Scan3 : scan edge (search edge 3) 4 [1979] 1979 root296Tail3 winner = winner :=
  (root296Step3).trans root296Scan4
theorem root296Scan2 : scan edge (search edge 3) 4 [1979] 1979 root296Tail2 winner = winner :=
  (root296Step2).trans root296Scan3
theorem root296Scan1 : scan edge (search edge 3) 4 [1979] 1979 root296Tail1 winner = winner :=
  (root296Step1).trans root296Scan2
theorem root296Scan0 : scan edge (search edge 3) 4 [1979] 1979 root296Tail0 winner = winner :=
  (root296Step0).trans root296Scan1
set_option maxRecDepth 4096 in
theorem root296Branch : search edge 4 [1979] root296Candidates 1979 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1979] 1979 root296Tail0 winner = winner
  exact root296Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
