import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root150Candidates : List Nat := Data.row883
set_option maxRecDepth 8192 in
theorem root150Candidates_checked : candidates edge 5 0 883 tail151 winner.bound = root150Candidates := by
  apply candidates_eq_row edge 5 0 883 tail151 root150Candidates winner.bound tail151_ordered
  · decide
  · decide
  · funext q
    rfl
def root150Tail57 : List Nat := []
def root150Tail56 : List Nat := 25087 :: root150Tail57
def root150Tail55 : List Nat := 24097 :: root150Tail56
def root150Tail54 : List Nat := 23629 :: root150Tail55
def root150Tail53 : List Nat := 22573 :: root150Tail54
def root150Tail52 : List Nat := 22543 :: root150Tail53
def root150Tail51 : List Nat := 22453 :: root150Tail52
def root150Tail50 : List Nat := 21433 :: root150Tail51
def root150Tail49 : List Nat := 21031 :: root150Tail50
def root150Tail48 : List Nat := 19801 :: root150Tail49
def root150Tail47 : List Nat := 19381 :: root150Tail48
def root150Tail46 : List Nat := 19219 :: root150Tail47
def root150Tail45 : List Nat := 18913 :: root150Tail46
def root150Tail44 : List Nat := 18679 :: root150Tail45
def root150Tail43 : List Nat := 18493 :: root150Tail44
def root150Tail42 : List Nat := 18169 :: root150Tail43
def root150Tail41 : List Nat := 18049 :: root150Tail42
def root150Tail40 : List Nat := 17881 :: root150Tail41
def root150Tail39 : List Nat := 17293 :: root150Tail40
def root150Tail38 : List Nat := 17167 :: root150Tail39
def root150Tail37 : List Nat := 16699 :: root150Tail38
def root150Tail36 : List Nat := 16567 :: root150Tail37
def root150Tail35 : List Nat := 15973 :: root150Tail36
def root150Tail34 : List Nat := 15217 :: root150Tail35
def root150Tail33 : List Nat := 14929 :: root150Tail34
def root150Tail32 : List Nat := 14797 :: root150Tail33
def root150Tail31 : List Nat := 14593 :: root150Tail32
def root150Tail30 : List Nat := 14431 :: root150Tail31
def root150Tail29 : List Nat := 14419 :: root150Tail30
def root150Tail28 : List Nat := 13807 :: root150Tail29
def root150Tail27 : List Nat := 13633 :: root150Tail28
def root150Tail26 : List Nat := 12919 :: root150Tail27
def root150Tail25 : List Nat := 11953 :: root150Tail26
def root150Tail24 : List Nat := 11779 :: root150Tail25
def root150Tail23 : List Nat := 11071 :: root150Tail24
def root150Tail22 : List Nat := 10939 :: root150Tail23
def root150Tail21 : List Nat := 10531 :: root150Tail22
def root150Tail20 : List Nat := 9967 :: root150Tail21
def root150Tail19 : List Nat := 9631 :: root150Tail20
def root150Tail18 : List Nat := 9601 :: root150Tail19
def root150Tail17 : List Nat := 9421 :: root150Tail18
def root150Tail16 : List Nat := 9043 :: root150Tail17
def root150Tail15 : List Nat := 9007 :: root150Tail16
def root150Tail14 : List Nat := 8677 :: root150Tail15
def root150Tail13 : List Nat := 7993 :: root150Tail14
def root150Tail12 : List Nat := 7753 :: root150Tail13
def root150Tail11 : List Nat := 6661 :: root150Tail12
def root150Tail10 : List Nat := 6571 :: root150Tail11
def root150Tail9 : List Nat := 6367 :: root150Tail10
def root150Tail8 : List Nat := 6151 :: root150Tail9
def root150Tail7 : List Nat := 6043 :: root150Tail8
def root150Tail6 : List Nat := 4561 :: root150Tail7
def root150Tail5 : List Nat := 3049 :: root150Tail6
def root150Tail4 : List Nat := 2749 :: root150Tail5
def root150Tail3 : List Nat := 2377 :: root150Tail4
def root150Tail2 : List Nat := 1987 :: root150Tail3
def root150Tail1 : List Nat := 1783 :: root150Tail2
def root150Tail0 : List Nat := 991 :: root150Tail1
def root150Candidates0 : List Nat := [18493]
set_option maxRecDepth 8192 in
theorem root150Candidates0_checked : candidates edge 4 883 991 root150Tail1 winner.bound = root150Candidates0 := by
  unfold candidates
  have row : edge 991 = (SortedNatTree.contains FastRows.row991) := by
    change Data.row991.contains = (SortedNatTree.contains FastRows.row991)
    exact FastRows.row991_checked.symm
  rw [row]
  decide
theorem root150Step0 : scan edge (search edge 3) 4 [883] 883 root150Tail0 winner =
    scan edge (search edge 3) 4 [883] 883 root150Tail1 winner := by
  rw [root150Tail0, scan, if_neg (by decide), root150Candidates0_checked]
  change scan edge (search edge 3) 4 [883] 883 root150Tail1
    (if 3 ≤ root150Candidates0.length then search edge 3 [883, 991] root150Candidates0 1874 winner else winner) = _
  rw [if_neg (by decide)]
def root150Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root150Candidates1_checked : candidates edge 4 883 1783 root150Tail2 winner.bound = root150Candidates1 := by
  unfold candidates
  have row : edge 1783 = (SortedNatTree.contains FastRows.row1783) := by
    change Data.row1783.contains = (SortedNatTree.contains FastRows.row1783)
    exact FastRows.row1783_checked.symm
  rw [row]
  decide
theorem root150Step1 : scan edge (search edge 3) 4 [883] 883 root150Tail1 winner =
    scan edge (search edge 3) 4 [883] 883 root150Tail2 winner := by
  rw [root150Tail1, scan, if_neg (by decide), root150Candidates1_checked]
  change scan edge (search edge 3) 4 [883] 883 root150Tail2
    (if 3 ≤ root150Candidates1.length then search edge 3 [883, 1783] root150Candidates1 2666 winner else winner) = _
  rw [if_neg (by decide)]
def root150Candidates2 : List Nat := [9421, 17293]
set_option maxRecDepth 8192 in
theorem root150Candidates2_checked : candidates edge 4 883 1987 root150Tail3 winner.bound = root150Candidates2 := by
  unfold candidates
  have row : edge 1987 = (SortedNatTree.contains FastRows.row1987) := by
    change Data.row1987.contains = (SortedNatTree.contains FastRows.row1987)
    exact FastRows.row1987_checked.symm
  rw [row]
  decide
theorem root150Step2 : scan edge (search edge 3) 4 [883] 883 root150Tail2 winner =
    scan edge (search edge 3) 4 [883] 883 root150Tail3 winner := by
  rw [root150Tail2, scan, if_neg (by decide), root150Candidates2_checked]
  change scan edge (search edge 3) 4 [883] 883 root150Tail3
    (if 3 ≤ root150Candidates2.length then search edge 3 [883, 1987] root150Candidates2 2870 winner else winner) = _
  rw [if_neg (by decide)]
def root150Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root150Candidates3_checked : candidates edge 4 883 2377 root150Tail4 winner.bound = root150Candidates3 := by
  unfold candidates
  have row : edge 2377 = (SortedNatTree.contains FastRows.row2377) := by
    change Data.row2377.contains = (SortedNatTree.contains FastRows.row2377)
    exact FastRows.row2377_checked.symm
  rw [row]
  decide
theorem root150Step3 : scan edge (search edge 3) 4 [883] 883 root150Tail3 winner =
    scan edge (search edge 3) 4 [883] 883 root150Tail4 winner := by
  rw [root150Tail3, scan, if_neg (by decide), root150Candidates3_checked]
  change scan edge (search edge 3) 4 [883] 883 root150Tail4
    (if 3 ≤ root150Candidates3.length then search edge 3 [883, 2377] root150Candidates3 3260 winner else winner) = _
  rw [if_neg (by decide)]
def root150Candidates4 : List Nat := []
set_option maxRecDepth 8192 in
theorem root150Candidates4_checked : candidates edge 4 883 2749 root150Tail5 winner.bound = root150Candidates4 := by
  unfold candidates
  have row : edge 2749 = (SortedNatTree.contains FastRows.row2749) := by
    change Data.row2749.contains = (SortedNatTree.contains FastRows.row2749)
    exact FastRows.row2749_checked.symm
  rw [row]
  decide
theorem root150Step4 : scan edge (search edge 3) 4 [883] 883 root150Tail4 winner =
    scan edge (search edge 3) 4 [883] 883 root150Tail5 winner := by
  rw [root150Tail4, scan, if_neg (by decide), root150Candidates4_checked]
  change scan edge (search edge 3) 4 [883] 883 root150Tail5
    (if 3 ≤ root150Candidates4.length then search edge 3 [883, 2749] root150Candidates4 3632 winner else winner) = _
  rw [if_neg (by decide)]
def root150Candidates5 : List Nat := []
set_option maxRecDepth 8192 in
theorem root150Candidates5_checked : candidates edge 4 883 3049 root150Tail6 winner.bound = root150Candidates5 := by
  unfold candidates
  have row : edge 3049 = (SortedNatTree.contains FastRows.row3049) := by
    change Data.row3049.contains = (SortedNatTree.contains FastRows.row3049)
    exact FastRows.row3049_checked.symm
  rw [row]
  decide
theorem root150Step5 : scan edge (search edge 3) 4 [883] 883 root150Tail5 winner =
    scan edge (search edge 3) 4 [883] 883 root150Tail6 winner := by
  rw [root150Tail5, scan, if_neg (by decide), root150Candidates5_checked]
  change scan edge (search edge 3) 4 [883] 883 root150Tail6
    (if 3 ≤ root150Candidates5.length then search edge 3 [883, 3049] root150Candidates5 3932 winner else winner) = _
  rw [if_neg (by decide)]
def root150Candidates6 : List Nat := []
set_option maxRecDepth 8192 in
theorem root150Candidates6_checked : candidates edge 4 883 4561 root150Tail7 winner.bound = root150Candidates6 := by
  unfold candidates
  have row : edge 4561 = (SortedNatTree.contains FastRows.row4561) := by
    change Data.row4561.contains = (SortedNatTree.contains FastRows.row4561)
    exact FastRows.row4561_checked.symm
  rw [row]
  decide
theorem root150Step6 : scan edge (search edge 3) 4 [883] 883 root150Tail6 winner =
    scan edge (search edge 3) 4 [883] 883 root150Tail7 winner := by
  rw [root150Tail6, scan, if_neg (by decide), root150Candidates6_checked]
  change scan edge (search edge 3) 4 [883] 883 root150Tail7
    (if 3 ≤ root150Candidates6.length then search edge 3 [883, 4561] root150Candidates6 5444 winner else winner) = _
  rw [if_neg (by decide)]
def root150Candidates7 : List Nat := []
set_option maxRecDepth 8192 in
theorem root150Candidates7_checked : candidates edge 4 883 6043 root150Tail8 winner.bound = root150Candidates7 := by
  unfold candidates
  have row : edge 6043 = (SortedNatTree.contains FastRows.row6043) := by
    change Data.row6043.contains = (SortedNatTree.contains FastRows.row6043)
    exact FastRows.row6043_checked.symm
  rw [row]
  decide
theorem root150Step7 : scan edge (search edge 3) 4 [883] 883 root150Tail7 winner =
    scan edge (search edge 3) 4 [883] 883 root150Tail8 winner := by
  rw [root150Tail7, scan, if_neg (by decide), root150Candidates7_checked]
  change scan edge (search edge 3) 4 [883] 883 root150Tail8
    (if 3 ≤ root150Candidates7.length then search edge 3 [883, 6043] root150Candidates7 6926 winner else winner) = _
  rw [if_neg (by decide)]
def root150Candidates8 : List Nat := [6571]
set_option maxRecDepth 8192 in
theorem root150Candidates8_checked : candidates edge 4 883 6151 root150Tail9 winner.bound = root150Candidates8 := by
  unfold candidates
  have row : edge 6151 = (SortedNatTree.contains FastRows.row6151) := by
    change Data.row6151.contains = (SortedNatTree.contains FastRows.row6151)
    exact FastRows.row6151_checked.symm
  rw [row]
  decide
theorem root150Step8 : scan edge (search edge 3) 4 [883] 883 root150Tail8 winner =
    scan edge (search edge 3) 4 [883] 883 root150Tail9 winner := by
  rw [root150Tail8, scan, if_neg (by decide), root150Candidates8_checked]
  change scan edge (search edge 3) 4 [883] 883 root150Tail9
    (if 3 ≤ root150Candidates8.length then search edge 3 [883, 6151] root150Candidates8 7034 winner else winner) = _
  rw [if_neg (by decide)]
theorem root150Scan9 : scan edge (search edge 3) 4 [883] 883 root150Tail9 winner = winner := by rw [root150Tail9, scan, if_pos (by decide)]
theorem root150Scan8 : scan edge (search edge 3) 4 [883] 883 root150Tail8 winner = winner :=
  (root150Step8).trans root150Scan9
theorem root150Scan7 : scan edge (search edge 3) 4 [883] 883 root150Tail7 winner = winner :=
  (root150Step7).trans root150Scan8
theorem root150Scan6 : scan edge (search edge 3) 4 [883] 883 root150Tail6 winner = winner :=
  (root150Step6).trans root150Scan7
theorem root150Scan5 : scan edge (search edge 3) 4 [883] 883 root150Tail5 winner = winner :=
  (root150Step5).trans root150Scan6
theorem root150Scan4 : scan edge (search edge 3) 4 [883] 883 root150Tail4 winner = winner :=
  (root150Step4).trans root150Scan5
theorem root150Scan3 : scan edge (search edge 3) 4 [883] 883 root150Tail3 winner = winner :=
  (root150Step3).trans root150Scan4
theorem root150Scan2 : scan edge (search edge 3) 4 [883] 883 root150Tail2 winner = winner :=
  (root150Step2).trans root150Scan3
theorem root150Scan1 : scan edge (search edge 3) 4 [883] 883 root150Tail1 winner = winner :=
  (root150Step1).trans root150Scan2
theorem root150Scan0 : scan edge (search edge 3) 4 [883] 883 root150Tail0 winner = winner :=
  (root150Step0).trans root150Scan1
set_option maxRecDepth 4096 in
theorem root150Branch : search edge 4 [883] root150Candidates 883 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [883] 883 root150Tail0 winner = winner
  exact root150Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
